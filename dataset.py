import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_basename(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_gt"):
        stem = stem[:-3]
    return stem


class ForgeryDataset(data.Dataset):
    def __init__(self, root: str, cfg: Dict, split: str = "train"):
        self.root = Path(root)
        self.cfg = cfg or {}
        self.split = split

        # Handle new split structure: root/train/ or root/test/
        split_dir = self.root / split
        if split_dir.exists():
            # New structure: dataset/CASIA2/train/ or dataset/CASIA2/test/ or dataset/IMD20/train/
            self.raw_dir = split_dir / "raw"
            self.mask_dir = split_dir / "mask"
            self.au_dir = split_dir / "au"  # May not exist for IMD20
            self.base_dir = split_dir
        else:
            # Fallback to old structure: dataset/CASIA2/raw/, etc.
            self.raw_dir = self.root / self.cfg.get("raw_dir", "raw")
            self.mask_dir = self.root / self.cfg.get("mask_dir", "mask")
            self.au_dir = self.root / self.cfg.get("au_dir", "Au")
            self.base_dir = self.root

        # Options
        self.include_au = bool(
            self.cfg.get("include_au", True)
        )  # Default True for new structure
        self.image_size = int(self.cfg.get("image_size", 320))
        self.crop_size = int(self.cfg.get("crop_size", self.image_size))
        self.use_center_crop_eval = bool(self.cfg.get("center_crop_eval", False))
        self.return_paths = bool(self.cfg.get("return_paths", False))
        self.extensions = tuple(
            self.cfg.get("extensions", [".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
        )

        # Optional split file (list of basenames without extensions)
        self.split_list: Optional[set] = None
        split_list_file = self.cfg.get(f"{split}_list")  # e.g., train_list, val_list
        if split_list_file is not None:
            split_list_path = Path(split_list_file)
            if split_list_path.is_file():
                items = [
                    line.strip()
                    for line in split_list_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.split_list = set(items)

        # Build pairs
        self.samples = self._build_samples()

        print(f"ForgeryDataset[{split}] -> {len(self.samples)} samples")
        if self.base_dir != self.root:
            print(f"  Base dir: {self.base_dir}")
        print(f"  Raw: {self.raw_dir} ({len(self._list_files(self.raw_dir))} files)")
        print(f"  Mask: {self.mask_dir} ({len(self._list_files(self.mask_dir))} files)")
        if self.include_au:
            print(f"  Au: {self.au_dir} ({len(self._list_files(self.au_dir))} files)")

    def _list_files(self, folder: Path) -> List[Path]:
        if not folder.exists():
            return []
        files = []
        for fp in folder.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in self.extensions:
                files.append(fp)
        return files

    def _build_samples(self) -> List[Tuple[Path, Optional[Path]]]:
        samples: List[Tuple[Path, Optional[Path]]] = []

        # Build tampered pairs (raw + mask) from current split
        if self.raw_dir.exists() and self.mask_dir.exists():
            # Index raw and mask by normalized basename
            raw_index: Dict[str, List[Path]] = {}
            for p in self._list_files(self.raw_dir):
                base = normalize_basename(p.name)
                if self.split_list and base not in self.split_list:
                    continue
                raw_index.setdefault(base, []).append(p)

            mask_index: Dict[str, List[Path]] = {}
            for p in self._list_files(self.mask_dir):
                base = normalize_basename(p.name)
                if self.split_list and base not in self.split_list:
                    continue
                mask_index.setdefault(base, []).append(p)

            # Create tampered pairs
            matched = sorted(set(raw_index.keys()) & set(mask_index.keys()))
            for base in matched:
                # Choose first match if multiple
                img_path = sorted(raw_index[base])[0]
                mask_path = sorted(mask_index[base])[0]
                samples.append((img_path, mask_path))

            print(f"  Found {len(matched)} tampered pairs")

        # Add authentic images
        if self.include_au and self.au_dir.exists():
            au_samples = 0
            for p in self._list_files(self.au_dir):
                base = normalize_basename(p.name)
                if self.split_list and base not in self.split_list:
                    continue
                # Add as negative sample (no mask)
                samples.append((p, None))
                au_samples += 1

            print(f"  Found {au_samples} authentic images")

        return samples

    def __len__(self):
        return len(self.samples)

    def _get_pair_transforms(self, is_train: bool):
        size = (self.image_size, self.image_size)
        crop = (self.crop_size, self.crop_size)

        def transform_pair(
            img: Image.Image, mask: Image.Image
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # Resize both
            img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)

            if is_train:
                # Random crop
                i, j, h, w = transforms.RandomCrop.get_params(img, output_size=crop)
                img = TF.crop(img, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
                # Random horizontal flip
                if random.random() < 0.5:
                    img = TF.hflip(img)
                    mask = TF.hflip(mask)
                # Random vertical flip (low prob)
                if random.random() < 0.1:
                    img = TF.vflip(img)
                    mask = TF.vflip(mask)
            else:
                if self.use_center_crop_eval and crop != size:
                    img = TF.center_crop(img, crop)
                    mask = TF.center_crop(mask, crop)

            img_t = TF.to_tensor(img)
            # Normalize to ImageNet stats by default (configurable later if needed)
            img_t = TF.normalize(
                img_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )

            # Mask to tensor in {0,1}
            mask_t = TF.to_tensor(mask)
            mask_t = (mask_t > 0.5).float()  # ensure binary
            # If mask has 3 channels, reduce to 1
            if mask_t.shape[0] > 1:
                mask_t = mask_t[0:1, ...]

            return img_t, mask_t

        return transform_pair

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if mask_path is None:
            # Authentic/negative sample: create zero mask same size as image
            mask = Image.new("L", img.size, color=0)
            sample_type = "authentic"
        else:
            # Tampered sample: load actual mask
            mask = Image.open(mask_path).convert("L")
            sample_type = "tampered"

        transform_pair = self._get_pair_transforms(is_train=(self.split == "train"))
        image_tensor, mask_tensor = transform_pair(img, mask)

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "filename": img_path.name,
            "is_tampered": bool(mask_tensor.max().item() > 0.5),
            "sample_type": sample_type,
            "has_mask": mask_path is not None,
        }

        if self.return_paths:
            sample["image_path"] = str(img_path)
            sample["mask_path"] = str(mask_path) if mask_path is not None else None

        return sample

    def data_collator(self, batch):
        """Custom collate function for batching."""
        valid_batch = [
            item
            for item in batch
            if item is not None and "image" in item and "mask" in item
        ]
        if not valid_batch:
            return None
        try:
            images = torch.stack([item["image"] for item in valid_batch])
            masks = torch.stack([item["mask"] for item in valid_batch])
            filenames = [item["filename"] for item in valid_batch]
            is_tampered = torch.tensor(
                [1 if item["is_tampered"] else 0 for item in valid_batch],
                dtype=torch.long,
            )
            sample_types = [item.get("sample_type", "unknown") for item in valid_batch]
            has_masks = torch.tensor(
                [1 if item.get("has_mask", False) else 0 for item in valid_batch],
                dtype=torch.long,
            )

            batch_out = {
                "images": images,
                "masks": masks,
                "filenames": filenames,
                "is_tampered": is_tampered,
                "sample_types": sample_types,
                "has_masks": has_masks,
            }
            if "image_path" in valid_batch[0]:
                batch_out["image_paths"] = [
                    item.get("image_path") for item in valid_batch
                ]
                batch_out["mask_paths"] = [
                    item.get("mask_path") for item in valid_batch
                ]
            return batch_out
        except Exception as e:
            print(f"Error in data collator: {e}")
            return None


def get_training_set(root, cfg):
    """Get training dataset"""
    return ForgeryDataset(root, cfg, split="train")


def get_test_set(root, cfg):
    """Get test/validation dataset"""
    return ForgeryDataset(root, cfg, split="test")


# Additional utility functions for data loading


def create_dataloader(
    dataset, batch_size, num_workers=4, shuffle=True, pin_memory=True
):
    """Create dataloader with proper error handling"""
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.data_collator,
        pin_memory=pin_memory,
        drop_last=shuffle,  # Drop last batch during training
        persistent_workers=num_workers > 0,
    )


def validate_dataset(dataset):
    """Validate dataset integrity"""
    print(f"Validating dataset with {len(dataset)} samples...")
    valid_samples = 0
    max_check = min(10, len(dataset))
    for i in range(max_check):
        try:
            sample = dataset[i]
            if (
                sample is not None
                and isinstance(sample.get("image"), torch.Tensor)
                and isinstance(sample.get("mask"), torch.Tensor)
            ):
                valid_samples += 1
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
    print(
        f"Dataset validation: {valid_samples}/{max_check} samples loaded successfully"
    )
    return valid_samples == max_check
