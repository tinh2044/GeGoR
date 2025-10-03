import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from fvcore.nn import FlopCountAnalysis

from skimage import exposure
from scipy import ndimage

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


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


def save_eval_images(
    inputs, pred_masks, targets, filenames, epoch, output_dir, save_all, origs=None
):
    """Save evaluation images with metrics"""
    eval_dir = Path(output_dir) / "evaluation" / f"epoch_{epoch}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if save_all:
        index = range(len(inputs))
    else:
        index = [np.random.randint(0, len(inputs))]
    for i in index:
        input_img = inputs[i].cpu()
        target_mask = targets[i].cpu()
        pred_mask = pred_masks[i].cpu()
        if origs is not None:
            orig_img = origs[i].cpu()
        filename = filenames[i]

        # Calculate metrics for this sample
        pred_binary = (pred_mask > 0.5).float()
        target_binary = (target_mask > 0.5).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        iou = (intersection / (union + 1e-8)).item()

        if origs is not None:
            j = 1
        else:
            j = 0

        fig, axes = plt.subplots(1, 4 + j, figsize=(20, 5))

        if origs is not None:
            # Original image
            orig_np = orig_img.permute(1, 2, 0).numpy()
            orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min())
            axes[0].imshow(orig_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

        input_np = input_img.permute(1, 2, 0).numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        axes[0 + j].imshow(input_np)
        axes[0 + j].set_title("Input Image")
        axes[0 + j].axis("off")

        # Ground truth mask
        target_np = target_mask.squeeze().numpy()
        axes[1 + j].imshow(target_np, cmap="gray")
        axes[1 + j].set_title("Ground Truth")
        axes[1 + j].axis("off")

        # Predicted mask
        pred_np = pred_mask.squeeze().numpy()
        axes[2 + j].imshow(pred_np, cmap="gray")
        axes[2 + j].set_title("Prediction")
        axes[2 + j].axis("off")

        # Overlay
        overlay = input_np.copy()
        overlay[target_np > 0.5] = [1, 0, 0]  # Red for ground truth
        overlay[pred_np > 0.5] = [0, 1, 0]  # Green for prediction
        axes[3 + j].imshow(overlay)
        axes[3 + j].set_title(f"Overlay (IoU: {iou:.3f})")
        axes[3 + j].axis("off")

        plt.suptitle(f"File: {filename}", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            eval_dir / f"eval_{i}_{Path(filename).stem}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """Convert a tensor (C,H,W) or (H,W) to numpy image in [0,1] range."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    arr = t.numpy()
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
        elif arr.shape[0] == 1:
            arr = arr[0]
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
        else:
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
    elif arr.ndim == 2:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
    return arr


def _quantize_array(arr: np.ndarray, levels: int = 256) -> np.ndarray:
    """Quantize array to discrete levels for pixelated effect.

    Args:
        arr: Input array in [0,1] range
        levels: Number of quantization levels

    Returns:
        Quantized array with discrete values
    """
    # Scale to desired levels and round to nearest integer
    quantized = np.round(arr * (levels - 1))
    # Normalize back to [0,1]
    quantized = quantized / (levels - 1)
    return quantized


def _pixelate_array(arr: np.ndarray, pixel_size: int = 4) -> np.ndarray:
    """Create pixelated effect by downsampling and upsampling.

    Args:
        arr: Input array in [0,1] range
        pixel_size: Size of each pixel block

    Returns:
        Pixelated array
    """
    if pixel_size <= 1:
        return arr

    # Get original shape
    H, W = arr.shape[:2]

    # Downsample by taking mean of blocks
    H_new = H // pixel_size
    W_new = W // pixel_size

    # Create downsampled version
    downsampled = np.zeros((H_new, W_new))
    for i in range(H_new):
        for j in range(W_new):
            block = arr[
                i * pixel_size : (i + 1) * pixel_size,
                j * pixel_size : (j + 1) * pixel_size,
            ]
            downsampled[i, j] = np.mean(block)

    # Upsample by repeating values
    pixelated = np.repeat(
        np.repeat(downsampled, pixel_size, axis=0), pixel_size, axis=1
    )

    # Crop to original size if needed
    return pixelated[:H, :W]


def _save_pixelated_image(
    arr: np.ndarray,
    path: str,
    pixel_style: str = "sharp",
    pixel_size: int = 4,
    quantize_levels: int = 64,
    dpi: int = 100,
):
    """Save array as pixelated image for better feature visualization.

    Args:
        arr: Array in [0,1] range
        path: Output path
        pixel_style: 'sharp', 'quantized', 'pixelated', 'smooth'
        pixel_size: Size for pixelation effect
        quantize_levels: Number of levels for quantization
        dpi: DPI for image quality
    """
    # Prepare array based on style
    if pixel_style == "quantized":
        arr = _quantize_array(arr, quantize_levels)
    elif pixel_style == "pixelated":
        arr = _pixelate_array(arr, pixel_size)
    elif pixel_style == "both":
        arr = _quantize_array(arr, quantize_levels)
        arr = _pixelate_array(arr, pixel_size)

    # Create figure with exact pixel dimensions
    fig, ax = plt.subplots(figsize=(arr.shape[1] / dpi, arr.shape[0] / dpi), dpi=dpi)

    # Turn off axes
    ax.axis("off")

    # Display with nearest neighbor interpolation
    if arr.ndim == 2:
        im = ax.imshow(arr, cmap="gray", interpolation="nearest")
    else:
        im = ax.imshow(arr, interpolation="nearest")

    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save with high quality
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def _resize_feature_map(feat: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize feature map to target shape using scipy.

    Args:
        feat: Feature map array (H, W) or (C, H, W)
        target_shape: Target shape (H, W)

    Returns:
        Resized feature map
    """
    if feat.ndim == 2:
        # Single channel
        return ndimage.zoom(
            feat,
            (target_shape[0] / feat.shape[0], target_shape[1] / feat.shape[1]),
            order=1,
        )
    elif feat.ndim == 3:
        # Multi-channel
        resized_channels = []
        for c in range(feat.shape[0]):
            resized = ndimage.zoom(
                feat[c],
                (target_shape[0] / feat.shape[1], target_shape[1] / feat.shape[2]),
                order=1,
            )
            resized_channels.append(resized)
        return np.stack(resized_channels, axis=0)
    else:
        return feat


def _save_enhanced_pixelated_image(
    arr: np.ndarray,
    path: str,
    enhancement: str = "clahe",
    colormap: str = "jet",
    pixel_style: str = "sharp",
    pixel_size: int = 1,
):
    """Save enhanced and pixelated feature map.

    Args:
        arr: Array in [0,1] range
        path: Output path
        enhancement: Enhancement method
        colormap: Color map
        pixel_style: Pixelation style
        pixel_size: Pixel size
    """
    # Apply enhancement
    enhanced = _enhance_feature_map(arr, enhancement)

    # Apply pixelation
    if pixel_style == "quantized":
        enhanced = _quantize_array(enhanced, 64)
    elif pixel_style == "pixelated":
        enhanced = _pixelate_array(enhanced, pixel_size)
    elif pixel_style == "both":
        enhanced = _quantize_array(enhanced, 64)
        enhanced = _pixelate_array(enhanced, pixel_size)

    # Apply colormap
    if enhanced.ndim == 2:
        colored = _apply_colormap(enhanced, colormap)
    else:
        colored = enhanced

    # Save with nearest neighbor interpolation
    fig, ax = plt.subplots(
        figsize=(colored.shape[1] / 100, colored.shape[0] / 100), dpi=100
    )
    ax.axis("off")

    im = ax.imshow(colored, interpolation="nearest")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


def _slice_tensor_to_image(t: torch.Tensor, b: int) -> list:
    """Slice a tensor at batch index b and convert to image-like numpy arrays.

    Returns a list of (suffix, np.ndarray) where suffix is appended to filename.
    Supported shapes after slicing to one sample:
      - (C,H,W) with C in {1,3} → single image
      - (C,H,W) with C>3       → per-channel images with suffix _chXXX
      - (H,W,2)                → magnitude map with suffix _mag
      - (H,W)                  → single image
      - (1,H,W)                → squeeze to (H,W)
    Other shapes are ignored as non image-like.
    """
    out: list[tuple[str, np.ndarray]] = []
    if not isinstance(t, torch.Tensor):
        return out
    try:
        x = t
        if x.dim() >= 1 and x.size(0) > b:
            # Slice batch dim if present
            if x.dim() >= 4:  # assume (B, ...)
                x = x[b].detach().cpu()
            elif x.dim() == 3 and x.size(0) == b:
                x = x.detach().cpu()
        else:
            x = x.detach().cpu()

        # Now interpret x
        if x.dim() == 3 and x.size(0) in (1, 3):  # (C,H,W)
            out.append(("", _to_numpy_image(x)))
        elif x.dim() == 3 and x.size(-1) == 2:  # (H,W,2)
            arr = x.numpy()
            mag = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2)
            mn, mx = mag.min(), mag.max()
            if mx > mn:
                mag = (mag - mn) / (mx - mn)
            out.append(("_mag", mag))
        elif x.dim() == 3:  # (C,H,W) with C>3 – per-channel
            C = x.size(0)
            for c in range(max(10, C)):
                out.append((f"_ch{c:03d}", _to_numpy_image(x[c : c + 1])))
        elif x.dim() == 2:  # (H,W)
            out.append(("", _to_numpy_image(x)))
        elif x.dim() == 3 and x.size(0) == 1:  # (1,H,W)
            out.append(("", _to_numpy_image(x[0])))
    except Exception:
        pass
    return out


def _enhance_feature_map(arr: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Enhance feature map for better visualization using various methods.

    Args:
        arr: numpy array in [0,1] range
        method: enhancement method ('clahe', 'histogram', 'gamma', 'none')
    """

    if method == "clahe":
        enhanced = exposure.equalize_adapthist(arr, clip_limit=0.03)
        return enhanced

    if method == "histogram":
        enhanced = exposure.equalize_hist(arr)
        return enhanced

    if method == "gamma":
        gamma = 0.5  # Darken to show more details
        enhanced = np.power(arr, gamma)
        return enhanced

    return arr


def _apply_colormap(arr: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Apply a colormap to convert grayscale to color image.

    Args:
        arr: numpy array in [0,1] range (H,W)
        colormap: matplotlib colormap name
    Returns:
        RGB image array (H,W,3) in [0,1] range
    """
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(colormap)

        colored = cmap(arr)

        if colored.shape[-1] == 4:
            colored = colored[..., :3]

        return colored
    except Exception:
        return np.stack([arr, arr, arr], axis=-1)


def recursion_save_feature(
    out_dir: Path,
    outputs,
    batch_index: int,
    target_shape=None,
    enhancement: str = "clahe",
    colormap: str = "jet",
    pixel_style: str = "pixelated",
    pixel_size: int = 4,
    log: callable = print,
    pbar=None,
):
    saved = 0
    for key, val in outputs.items():
        subdir = out_dir / key
        if isinstance(val, dict):
            saved += recursion_save_feature(
                subdir,
                val,
                batch_index=batch_index,
                target_shape=target_shape,
                enhancement=enhancement,
                colormap=colormap,
                pixel_style=pixel_style,
                pixel_size=pixel_size,
                log=log,
                pbar=pbar,
            )
            continue
        if not isinstance(val, torch.Tensor):
            continue

        # Try to get image arrays for this batch index
        images = _slice_tensor_to_image(val, batch_index)
        if not images:
            continue

        subdir.mkdir(parents=True, exist_ok=True)
        for suf, arr in images:
            if arr.ndim == 2 and target_shape is not None:
                arr = _resize_feature_map(arr, target_shape)
            save_path = subdir / f"{key}{suf}.png"
            _save_enhanced_pixelated_image(
                arr,
                str(save_path),
                enhancement=enhancement,
                colormap=colormap,
                pixel_style=pixel_style,
                pixel_size=pixel_size,
            )
            saved += 1
            if pbar is not None:
                try:
                    pbar.update(1)
                except Exception:
                    pass
    if saved and log is not None:
        log(f"Saved {saved} image(s) under {out_dir}")
    return saved


def _estimate_image_count(outputs, batch_index: int) -> int:
    count = 0
    for _, val in outputs.items():
        if isinstance(val, dict):
            count += _estimate_image_count(val, batch_index)
        elif isinstance(val, torch.Tensor):
            try:
                imgs = _slice_tensor_to_image(val, batch_index)
                count += len(max(10, val.shape[0]))
            except Exception:
                pass
    return count


def save_features_per_channel(
    inputs: torch.Tensor,
    pred_masks: torch.Tensor,
    targets: torch.Tensor,
    outputs: dict,
    filenames,
    epoch: int,
    output_dir,
    enhancement: str = "clahe",
    colormap: str = "jet",
    pixel_style: str = "pixelated",
    pixel_size: int = 4,
    resize_to_target: bool = True,
    use_tqdm: bool = True,
):
    base = Path(output_dir) / "features"
    base.mkdir(parents=True, exist_ok=True)

    B = inputs.size(0)
    if not isinstance(filenames, (list, tuple)):
        filenames = [str(filenames)] * B

    for i in range(B):
        name = Path(str(filenames[i])).stem
        out_dir = base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        in_img = _to_numpy_image(inputs[i])
        plt.imsave(out_dir / "input.png", in_img, cmap=None)

        # Get target shape for resizing
        target_shape = None
        if resize_to_target:
            tgt = targets[i]
            if tgt.ndim == 3 and tgt.shape[0] == 1:
                target_shape = (tgt.shape[1], tgt.shape[2])  # (H, W)
            elif tgt.ndim == 2:
                target_shape = (tgt.shape[0], tgt.shape[1])  # (H, W)

        tgt = targets[i]
        tgt_np = _to_numpy_image(
            tgt.squeeze(0) if tgt.ndim == 3 and tgt.shape[0] == 1 else tgt
        )
        plt.imsave(out_dir / "target.png", tgt_np, cmap="gray")

        pred = pred_masks[i]
        pred_np = _to_numpy_image(
            pred.squeeze(0) if pred.ndim == 3 and pred.shape[0] == 1 else pred
        )
        plt.imsave(out_dir / "pred.png", pred_np, cmap="gray")

        # Log header for this sample
        print(f"[features] Saving for sample {i} ({name}) ...")

        # Create tqdm progress bar if enabled
        pbar = None
        if use_tqdm and _HAS_TQDM:
            try:
                total_est = _estimate_image_count(outputs, i)
            except Exception:
                total_est = None
            pbar = tqdm(
                total=total_est, desc=f"features:{name}", unit="img", leave=False
            )

        # Recursively walk all outputs and save what looks like images for this sample
        total_saved = recursion_save_feature(
            out_dir,
            outputs,
            batch_index=i,
            target_shape=target_shape,
            enhancement=enhancement,
            colormap=colormap,
            pixel_style=pixel_style,
            pixel_size=pixel_size,
            log=lambda msg: print(f"[features][{name}] {msg}"),
            pbar=pbar,
        )

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

        print(f"[features] Done {name}: {total_saved} image(s) saved.")


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_rank():
    import torch.distributed as dist

    if not torch.distributed.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(obj, path):
    if is_main_process():
        from pathlib import Path as _Path

        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, p)


def setup_distributed():
    """Initialize distributed training (torchrun-friendly)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(
            os.environ.get(
                "LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
            )
        )

        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        init_method = os.environ.get("DIST_INIT_METHOD", "env://")

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method=init_method)

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        print(
            f"Setting up distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}, backend={backend}"
        )

        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


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

    try:
        from fvcore.nn import FlopCountAnalysis

        dummy_input = torch.randn(input_shape).to(device)

        flops = FlopCountAnalysis(model, dummy_input)
        flops_total = flops.total()
        flops_str = f"{flops_total / 1e9:.3f} GFLOPs"
        info.update(
            {
                "flops": int(flops_total),
                "flops_str": flops_str,
                "params_str": f"{total_params:,.3f}",
            }
        )
    except Exception:
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
