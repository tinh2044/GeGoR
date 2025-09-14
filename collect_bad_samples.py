import argparse
import shutil
from pathlib import Path
from typing import List

import torch
import time

from dataset import ForgeryDataset
from metrics import calculate_iou
from net import CMFDNet


def parse_args():
    parser = argparse.ArgumentParser("Collect samples with IoU below a threshold")
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to a yaml config file (e.g., configs/casiav2.yaml)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained checkpoint .pth (contains model_state_dict)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Which split to scan (default: test)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="IoU threshold; samples with IoU < threshold are copied",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="bad_sample",
        help="Output directory to copy bad samples",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Num workers for dataloader"
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=1,
        help="Print progress every N samples (default: 1)",
    )
    return parser.parse_args()


def load_config(cfg_path: Path) -> dict:
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def build_model(cfg_model: dict, ckpt_path: Path, device: torch.device) -> CMFDNet:
    model = CMFDNet(**cfg_model)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def ensure_output_dirs(base_out: Path):
    (base_out / "raw").mkdir(parents=True, exist_ok=True)
    (base_out / "mask").mkdir(parents=True, exist_ok=True)


def copy_pair(raw_path: Path, mask_path: Path, out_dir: Path):
    ensure_output_dirs(out_dir)
    shutil.copy2(raw_path, out_dir / "raw" / raw_path.name)
    shutil.copy2(mask_path, out_dir / "mask" / mask_path.name)


@torch.no_grad()
def evaluate_and_collect(
    cfg_data: dict,
    cfg_model: dict,
    ckpt_path: Path,
    split: str,
    iou_threshold: float,
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    print_freq: int,
):
    # We need the dataset to return paths so we can copy later
    cfg_data = dict(cfg_data)
    cfg_data["return_paths"] = True

    dataset = ForgeryDataset(cfg=cfg_data, split=split)
    total_samples = len(dataset)
    print(f"Scanning {total_samples} samples from split '{split}'...")

    # Find the dataset split folder to resolve raw/mask paths
    split_dir = Path(cfg_data["root"]) / cfg_data[f"{split}_dir"]
    raw_dir = split_dir / "raw"
    mask_dir = split_dir / "mask"

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.data_collator,
        pin_memory=True,
    )

    model = build_model(cfg_model, ckpt_path, device)

    num_bad = 0
    total = 0
    processed = 0
    last_end = time.time()
    iter_total = 0.0
    data_total = 0.0
    for batch in loader:
        if batch is None:
            continue
        # Data loading time since last iteration
        data_time = time.time() - last_end
        data_total += data_time
        last_end = time.time()
        images = batch.get("images").to(device)
        masks = batch.get("masks").to(device)
        filenames: List[str] = batch.get("filenames", [])
        image_paths: List[str] = batch.get("image_paths", [])
        mask_paths: List[str] = batch.get("mask_paths", [])

        outputs = model(images, gt_mask=masks)
        pred_masks = outputs["mask"]  # (B,1,H,W), probs

        # Compute IoU per sample in the batch
        # calculate_iou expects (B,H,W) or (B,1,H,W); it handles squeezing
        # We'll compute per-sample by slicing
        B = pred_masks.size(0)
        for i in range(B):
            pred_i = pred_masks[i : i + 1]  # keep batch dim
            gt_i = masks[i : i + 1]
            iou = calculate_iou(pred_i, gt_i, threshold=0.5)
            total += 1
            processed += 1
            if iou < iou_threshold:
                # Prefer exact original paths provided by the dataset
                raw_path = Path(image_paths[i]) if image_paths else None
                mask_path = Path(mask_paths[i]) if mask_paths else None

                if raw_path is None or mask_path is None:
                    name = filenames[i]
                    raw_path = raw_dir / name
                    mask_path = mask_dir / name
                    # If extensions or suffixes differ, resolve by stem (try with and without _gt)
                    if not raw_path.exists() or not mask_path.exists():
                        stem = Path(name).stem
                        cand_stems = [stem, stem + "_gt"]
                        raw_candidates = []
                        mask_candidates = []
                        for st in cand_stems:
                            raw_candidates.extend(list(raw_dir.rglob(st + ".*")))
                            mask_candidates.extend(list(mask_dir.rglob(st + ".*")))
                        if raw_candidates:
                            raw_path = sorted(raw_candidates)[0]
                        if mask_candidates:
                            mask_path = sorted(mask_candidates)[0]

                if raw_path and mask_path and raw_path.exists() and mask_path.exists():
                    copy_pair(raw_path, mask_path, out_dir)
                    num_bad += 1
                else:
                    missing = filenames[i] if i < len(filenames) else str(i)
                    print(f"Warning: cannot resolve paths for {missing}")

            # Iteration timing and progress logging
            iter_time = time.time() - last_end
            iter_total += iter_time
            last_end = time.time()

            if processed % max(1, print_freq) == 0 or processed == total_samples:
                avg_iter = iter_total / max(1, processed)
                eta_seconds = max(0.0, avg_iter * (total_samples - processed))
                eta_string = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                avg_data = data_total / max(1, processed)
                if torch.cuda.is_available() and device.type == "cuda":
                    mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    print(
                        f"Step [{processed}/{total_samples}], ETA: {eta_string}, time: {avg_iter:.4f}, data: {avg_data:.4f}, max mem: {mb:.0f}MB, bad: {num_bad}"
                    )
                else:
                    print(
                        f"Step [{processed}/{total_samples}], ETA: {eta_string}, time: {avg_iter:.4f}, data: {avg_data:.4f}, bad: {num_bad}"
                    )

    print(
        f"Done. {num_bad}/{total} samples (IoU < {iou_threshold}) copied to {out_dir}."
    )


def main():
    args = parse_args()
    cfg = load_config(Path(args.cfg))

    cfg_data = cfg["data"]
    cfg_model = cfg["model"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluate_and_collect(
        cfg_data=cfg_data,
        cfg_model=cfg_model,
        ckpt_path=Path(args.ckpt),
        split=args.split,
        iou_threshold=args.threshold,
        device=device,
        out_dir=out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        print_freq=args.print_freq,
    )


if __name__ == "__main__":
    main()
