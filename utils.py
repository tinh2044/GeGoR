from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F
import torch
from torchvision import transforms
import numpy as np
import cv2
import skimage
from skimage import transform

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from pathlib import Path

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def fscore(T_score):
    Fp, Fn, Tp = T_score[1:]
    f_score = 2 * Tp / (2 * Tp + Fp + Fn)
    return f_score


def precision(T_score):
    Fp, _, Tp = T_score[1:]
    recall = Tp / (Tp + Fp + 1e-8)
    return recall


def recall(T_score):
    _, Fn, Tp = T_score[1:]
    recall = Tp / (Tp + Fn + 1e-8)
    return recall


def conf_mat(labels, preds):
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels).ravel()
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds).ravel()

    tp = np.sum(preds[labels == 1] == 1)
    tn = np.sum(preds[labels == 0] == 0)
    fp = np.sum(preds[labels == 0] == 1)
    fn = np.sum(preds[labels == 1] == 0)
    return np.array([tn, fp, fn, tp])


class ForgeryTransform:
    """Data augmentation for forgery detection"""

    def __init__(self, size=(320, 320)):
        if isinstance(size, int) or isinstance(size, float):
            size = (size, size)

        # Geometric transformations
        self.scale = np.random.choice(np.linspace(0.8, 1.2, 30))
        self.translate = (
            np.random.choice(np.linspace(-0.1 * size[1], 0.1 * size[1], 50)),
            np.random.choice(np.linspace(-0.1 * size[0], 0.1 * size[0], 50)),
        )
        self.flip = np.random.rand() > 0.5
        self.rotate = np.random.choice(np.linspace(-15, 15, 30))

        self.tfm = transform.SimilarityTransform(
            scale=self.scale, translation=self.translate
        )

    def __call__(self, im=None, mask=None):
        if im is not None:
            # Apply geometric transformations
            im = transform.warp(im, self.tfm)
            if self.rotate != 0:
                im = transform.rotate(im, self.rotate, mode="reflect")
            if self.flip:
                im = np.flip(im, 1).copy()

        if mask is not None:
            # Apply same transformations to mask
            mask = transform.warp(mask, self.tfm)
            if self.rotate != 0:
                mask = transform.rotate(mask, self.rotate, mode="reflect")
            if self.flip:
                mask = np.flip(mask, 1).copy()
            # Ensure binary mask
            mask = (mask > 0.5).astype(np.float32)

        return im, mask


class CustomTransform:
    def __init__(self, size=224):
        if isinstance(size, int) or isinstance(size, float):
            self.size = (size, size)
        else:
            self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def resize(self, img=None, mask=None):
        if img is not None:
            img = skimage.img_as_float32(img)
            if img.shape[0] != self.size[0] or img.shape[1] != self.size[1]:
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            if mask.shape[0] != self.size[0] or mask.shape[1] != self.size[1]:
                mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return img, mask

    def inverse(self, x, mask=False):
        if x.is_cuda:
            x = x.squeeze().data.cpu().numpy()
        else:
            x = x.squeeze().data.numpy()
        x = x.transpose((1, 2, 0))
        if not mask:
            x = x * self.std + self.mean
        return x

    def __call__(self, img=None, mask=None, other_tfm=None):
        img, mask = self.resize(img, mask)
        if other_tfm is not None:
            img, mask = other_tfm(img, mask)
        if img is not None:
            img = (img - self.mean) / self.std
            img = self.to_tensor(img).float()

        if mask is not None:
            mask = self.to_tensor(mask).float()

        return img, mask


def custom_transform_images(images=None, masks=None, size=224, other_tfm=None):
    tsfm = CustomTransform(size=size)
    X, Y = None, None
    if images is not None:
        X = torch.zeros((images.shape[0], 3, size, size), dtype=torch.float32)
        for i in range(images.shape[0]):
            X[i] = tsfm(img=images[i], other_tfm=other_tfm)
    if masks is not None:
        Y = torch.zeros((masks.shape[0], 1, size, size), dtype=torch.float32)
        for i in range(masks.shape[0]):
            _, Y[i, 0] = tsfm(img=None, mask=masks[i], other_tfm=other_tfm)

    return X, Y


def add_overlay(im, m1, m2=None, alpha=0.5, c1=[0, 1, 0], c2=[1, 0, 0]):
    r, c = im.shape[:2]

    M1 = np.zeros((r, c, 3), dtype=np.float32)
    M2 = np.zeros((r, c, 3), dtype=np.float32)

    if m2 is not None:
        M1[m1 > 0] = c1
        M2[m2 > 0] = c2
        M = cv2.addWeighted(M1, alpha, M2, 1 - alpha, 0, None)
    else:
        M1[m1 > 0] = c1
        M = M1

    I = cv2.addWeighted(im, alpha, M, 1 - alpha, 0, None)

    return I


class MultiPagePdf:
    def __init__(self, total_im, out_name, nrows=4, ncols=4, figsize=(8, 6)):
        """init

        Keyword Arguments:
            total_im {int} -- #images
            nrows {int} -- #rows per page (default: {4})
            ncols {int} -- #columns per page (default: {4})
            figsize {tuple} -- fig size (default: {(8, 6)})
        """
        self.total_im = total_im
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = tuple(figsize)
        self.out_name = out_name

        # create figure and axes
        total_pages = int(np.ceil(total_im / (nrows * ncols)))

        self.figs = []
        self.axes = []

        for _ in range(total_pages):
            f, a = plt.subplots(nrows, ncols)

            f.set_size_inches(figsize)
            self.figs.append(f)
            self.axes.extend(a.flatten())

        self.cnt_ax = 0

    def plot_one(self, x, *args, **kwargs):
        ax = self.axes[self.cnt_ax]
        ax.imshow(x, *args, **kwargs)  # prediction
        # ax.imshow(x[0])  # ground truth

        ax.set_xticks([])
        ax.set_yticks([])

        self.cnt_ax += 1
        return ax

    def final(self):
        with PdfPages(self.out_name) as pdf:
            for fig in self.figs:
                fig.tight_layout()
                pdf.savefig(fig)
        plt.close("all")


class MMetric:
    def __init__(self, name=""):
        self.T = np.zeros(4)
        self.fscore = []
        self.prec = []
        self.rec = []
        self.name = name

    def update(self, gt, pred, batch_mode=True, log=True):
        if batch_mode:
            num = gt.shape[0]
            arr = []
            for i in range(num):
                fs = self._update(gt[i], pred[i], log=False)
                if fs != -1:
                    arr.append(fs)
            if len(arr) > 0:
                print(f"\t\t{self.name} f score : {np.mean(arr):.4f}")
        else:
            self._update(gt, pred, log=log)

    def _update(self, gt, pred, log=True):
        gt = gt.ravel()
        pred = pred.ravel()

        tt = conf_mat(gt, pred)
        self.T += tt

        if np.all(gt == 0):
            return -1

        prec = precision(tt)
        rec = recall(tt)
        fs = fscore(tt)

        self.prec.append(prec)
        self.rec.append(rec)
        self.fscore.append(fs)

        if log:
            print(
                f"{self.name} precision : {prec:.4f}, recall : {rec:.4f}, f1 : {fs:.4f}"
            )
        return fs

    def final(self):
        # protocal A
        print(f"\n{self.name} ")
        print("-" * 50)
        # print("\nProtocol A:")
        # print(
        #     f"precision : {precision(self.T):.4f}, recall : {recall(self.T):.4f}, f1 : {fscore(self.T):.4f}")

        # protocol B
        print("\nProtocol B:")
        print(
            f"precision : {np.mean(self.prec):.4f}, recall : {np.mean(self.rec):.4f}, f1 : {np.mean(self.fscore):.4f}"
        )

        return np.mean(self.fscore)


class Metric:
    def __init__(self, dims=3, names=["forge", "source", "pristine"]):
        self.names = names
        self.dims = dims
        assert len(names) == dims

        self.list_metrics = []
        for i in range(dims):
            self.list_metrics.append(MMetric(name=names[i]))

    def update(self, gt, pred, batch_mode=True):
        ind_gt = np.argmax(gt, axis=-3)
        ind_pred = np.argmax(pred, axis=-3)
        for i in range(self.dims):
            self.list_metrics[i].update(
                ind_gt == i, ind_pred == i, batch_mode=batch_mode
            )

    def final(self):
        sc = []
        for i in range(self.dims):
            sc.append(self.list_metrics[i].final())
        return np.mean(sc)


import collections


class Metric_image(object):
    def __init__(self):
        self.gt = []
        self.pred = []

    def update(self, _gt, _pred, thres=0.5):
        _gt = _gt > thres
        _pred = _pred > thres

        if isinstance(_gt, collections.abc.Iterable):
            self.gt.extend(list(_gt))
            self.pred.extend(list(_pred))
        else:
            self.gt.append(_gt)
            self.pred.append(_pred)

    def final(self):
        pr, re, f, _ = precision_recall_fscore_support(
            self.gt, self.pred, average="binary"
        )

        print("Image level score")
        print(f"precision: {pr:.4f}, recall: {re:.4f}, f-score: {f:.4f} ")


class ImAug:
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.50, aug)

        self.seq_for_coco_back = iaa.Sequential(
            [
                sometimes(
                    iaa.Affine(rotate=(-45, 45))
                ),  # rotate by -45 to 45 degrees (affects segmaps)
                # sometimes(iaa.ElasticTransformation(alpha=50, sigma=5)),  # apply water effect (affects segmaps)
                # sometimes(iaa.Fliplr(0.10)), # horizontally flip 10% of the images
            ],
            random_order=True,
        )

    @staticmethod
    def apply_contrast(im):
        seq = iaa.ContrastNormalization((0.75, 1.5))
        im = seq(images=skimage.img_as_ubyte(im))
        im = skimage.img_as_float32(im)
        return im

    @staticmethod
    def apply_water(mask):
        seq = iaa.ElasticTransformation(alpha=50, sigma=5)
        segmap = SegmentationMapsOnImage(
            skimage.img_as_ubyte(mask)[..., None], shape=mask.shape
        )
        mask = seq(segmentation_maps=segmap)
        mask = skimage.img_as_float32(mask.get_arr().squeeze())
        return mask

    def apply_coco_back(self, im, mask):
        im = skimage.img_as_ubyte(im)
        mask = skimage.img_as_ubyte(mask)[..., None]
        segmap = SegmentationMapsOnImage(mask, shape=im.shape)
        images_aug_i, segmaps_aug_i = self.seq_for_coco_back(
            image=im, segmentation_maps=segmap
        )
        im = skimage.img_as_float32(images_aug_i)
        mask = skimage.img_as_float32(segmaps_aug_i.get_arr().squeeze())
        return im, mask


def patch_transform(im_mask, mask_bb, new_centroid, translate=None, scale=None):
    if len(mask_bb) < 4:
        return im_mask.copy()

    patch_mask = im_mask[mask_bb[1] : mask_bb[3], mask_bb[0] : mask_bb[2]]
    resized_patch = cv2.resize(
        patch_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )

    hp, wp = resized_patch.shape[:2]

    topx = int(max(0, new_centroid[0] - wp / 2))
    topy = int(max(0, new_centroid[1] - hp / 2))

    bottomx = int(min(topx + wp, im_mask.shape[1]))
    bottomy = int(min(topy + hp, im_mask.shape[0]))

    w, h = bottomx - topx, bottomy - topy

    new_mask = np.zeros(im_mask.shape, dtype=patch_mask.dtype)

    new_mask[topy : topy + h, topx : topx + w] = resized_patch[:h, :w]

    return new_mask


def splice(img_target, img_source, img_mask):
    if img_target.shape != img_source.shape:
        img_target = skimage.transform.resize(
            img_target, img_mask.shape[:2], anti_aliasing=False, mode="reflect"
        )
        # img_target = skimage.img_as_ubyte(img_target)

    if len(img_mask.shape) < 3:
        img_mask = img_mask[..., None]

    img_mask = img_mask > 0
    if img_mask.dtype != np.float32:
        img_mask = img_mask.astype(np.float32)
    img_mani = img_mask * img_source + img_target * (1 - img_mask)
    # img_mani = img_mani.astype(np.uint8)

    return img_mani


# Utility functions for training and evaluation
def save_img(tensor, path, normalize=True):
    """Save tensor as image file"""
    if isinstance(tensor, torch.Tensor):
        # Convert tensor to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        img = tensor.detach().numpy()

        # Handle different tensor shapes
        if img.ndim == 4:  # Batch of images
            img = img[0]  # Take first image
        if img.ndim == 3:  # CHW format
            if img.shape[0] == 1:  # Single channel (mask)
                img = img[0]  # HW
            elif img.shape[0] == 3:  # RGB
                img = img.transpose(1, 2, 0)  # HWC
            else:
                raise ValueError(f"Unsupported channel number: {img.shape[0]}")
        elif img.ndim == 2:  # HW format (mask)
            pass
        else:
            raise ValueError(f"Unsupported tensor dimensions: {img.ndim}")

    # Normalize if needed
    if normalize and img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save image
    if len(img.shape) == 2:  # Grayscale
        plt.imsave(path, img, cmap="gray")
    else:  # RGB
        plt.imsave(path, img)


def save_sample_images(inputs, pred_masks, targets, batch_idx, epoch, output_dir):
    """Save sample images during training for visualization"""
    sample_dir = Path(output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors to images
    input_img = inputs[0].cpu()  # (3, H, W)
    target_mask = targets[0].cpu()  # (1, H, W)
    pred_mask = pred_masks[0].cpu()  # (1, H, W)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input image
    input_np = input_img.permute(1, 2, 0).numpy()
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
    axes[0].imshow(input_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Ground truth mask
    target_np = target_mask.squeeze().numpy()
    axes[1].imshow(target_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Predicted mask
    pred_np = (pred_mask.squeeze().detach().numpy() > 0.5).astype(float)
    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(
        sample_dir / f"epoch_{epoch}_batch_{batch_idx}.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def save_eval_images(inputs, pred_masks, targets, filenames, epoch, output_dir):
    """Save evaluation images with metrics"""
    eval_dir = Path(output_dir) / "evaluation" / f"epoch_{epoch}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(len(inputs), 5)):  # Save first 5 images
        input_img = inputs[i].cpu()
        target_mask = targets[i].cpu()
        pred_mask = pred_masks[i].cpu()
        filename = filenames[i]

        # Calculate metrics for this sample
        pred_binary = (pred_mask > 0.5).float()
        target_binary = (target_mask > 0.5).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        iou = (intersection / (union + 1e-8)).item()

        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Input image
        input_np = input_img.permute(1, 2, 0).numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        axes[0].imshow(input_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Ground truth mask
        target_np = target_mask.squeeze().numpy()
        axes[1].imshow(target_np, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # Predicted mask
        pred_np = pred_mask.squeeze().numpy()
        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        # Overlay
        overlay = input_np.copy()
        overlay[target_np > 0.5] = [1, 0, 0]  # Red for ground truth
        overlay[pred_np > 0.5] = [0, 1, 0]  # Green for prediction
        axes[3].imshow(overlay)
        axes[3].set_title(f"Overlay (IoU: {iou:.3f})")
        axes[3].axis("off")

        plt.suptitle(f"File: {filename}", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            eval_dir / f"eval_{i}_{Path(filename).stem}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for binary segmentation"""
    pred = pred.sigmoid()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    pred = pred.sigmoid()
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.exp(-bce)
    focal_loss = alpha * (1 - pt) ** gamma * bce
    return focal_loss.mean()


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def combined_loss(pred, target, weights=None):
    """Combined loss with BCE, Dice, and Focal losses"""
    if weights is None:
        weights = {"bce": 1.0, "dice": 1.0, "focal": 0.5}

    bce_loss = F.binary_cross_entropy_with_logits(pred, target.float())
    dice_loss_val = dice_loss(pred, target)
    focal_loss_val = focal_loss(pred, target)

    total_loss = (
        weights["bce"] * bce_loss
        + weights["dice"] * dice_loss_val
        + weights["focal"] * focal_loss_val
    )

    return total_loss


# ==========================
# Distributed utils and misc
# ==========================
def get_rank():
    """Return current process rank, 0 if not distributed."""
    import torch.distributed as dist  # local import to avoid hard dependency at import time

    if not torch.distributed.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(obj, path):
    """torch.save only on main process."""
    if is_main_process():
        from pathlib import Path as _Path

        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, p)


def setup_for_distributed(is_master: bool):
    """Disables printing when not in master process"""
    import builtins as __builtin__

    print_fn = __builtin__.print

    def print_if_master(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            print_fn(*args, **kwargs)

    __builtin__.print = print_if_master


def init_distributed_mode(args):
    """Initialize torch.distributed if requested by env/args.

    Works even if running on a single process; in that case it becomes a no-op.
    """
    import os as _os
    import platform as _platform
    import torch.distributed as dist

    if ("RANK" in _os.environ and "WORLD_SIZE" in _os.environ) or getattr(
        args, "world_size", 1
    ) > 1:
        args.rank = int(_os.environ.get("RANK", "0"))
        args.world_size = int(
            _os.environ.get("WORLD_SIZE", str(getattr(args, "world_size", 1)))
        )
        args.gpu = int(
            _os.environ.get("LOCAL_RANK", str(getattr(args, "local_rank", 0)))
        )
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = getattr(args, "local_rank", 0)

    args.distributed = args.world_size > 1

    if not args.distributed:
        setup_for_distributed(True)
        return

    backend = "nccl"
    # Windows or CPU-only should use gloo
    if _platform.system().lower().startswith("win") or not torch.cuda.is_available():
        backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=backend,
        init_method=getattr(args, "dist_url", "env://"),
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def count_model_parameters(model) -> int:
    """Count total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def get_model_info(model, input_shape, device) -> dict:
    """Return model parameter counts and, if possible, FLOPs and MACs.

    input_shape: tuple like (batch, channels, height, width)
    """
    info = {}
    total_params = count_model_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["total_params"] = total_params
    info["trainable_params"] = trainable_params
    info["non_trainable_params"] = total_params - trainable_params

    # Try to compute FLOPs and MACs using thop if available
    try:
        from thop import profile, clever_format

        dummy_input = torch.randn(*input_shape, device=device)
        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            flops, macs = profile(model, inputs=(dummy_input,), verbose=False)
        if model_was_training:
            model.train()
        flops_str, macs_str = clever_format([flops, macs], "%.3f")
        info.update(
            {
                "flops": int(flops),
                "macs": int(macs),
                "flops_str": flops_str,
                "macs_str": macs_str,
                "params_str": clever_format([total_params], "%.3f")[0],
            }
        )
    except Exception:
        # thop not available or failed; ignore
        pass

    return info


def check_state_dict(model, state_dict: dict) -> bool:
    """Basic compatibility check between a model and a checkpoint state_dict.

    Returns True if all keys in checkpoint exist in the model and shapes match
    for the overlapping keys.
    """
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())

    if not ckpt_keys:
        return False

    if not ckpt_keys.issubset(model_keys):
        return False

    for k in ckpt_keys:
        if k in model_state and model_state[k].shape != state_dict[k].shape:
            return False
    return True
