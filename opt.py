import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any

from metrics import compute_metrics
from utils import save_eval_images, save_sample_images, dice_loss, focal_loss
from logger import MetricLogger, SmoothedValue


def train_one_epoch(
    args, model, data_loader, optimizer, scheduler, epoch, print_freq=10, log_dir="logs"
):
    """Train for one epoch - forgery detection"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Train epoch: [{epoch}]"

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # Handle different batch formats
        if isinstance(batch, dict):
            inputs = batch.get("images", batch.get("inputs")).to(args.device)
            targets = batch.get("masks", batch.get("targets")).to(args.device)
            filenames = batch.get(
                "filenames", [f"sample_{batch_idx}_{i}" for i in range(inputs.size(0))]
            )
        else:
            # Legacy format
            inputs, targets = batch[0].to(args.device), batch[1].to(args.device)
            filenames = [f"sample_{batch_idx}_{i}" for i in range(inputs.size(0))]

        outputs = model(inputs, gt_mask=targets)
        if isinstance(outputs, dict):
            pred_masks = outputs.get("mask", outputs.get("output", None))
            loss_dict = outputs.get("loss", None)
        else:
            pred_masks = outputs

        total_loss = loss_dict["total"]

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # Note: scheduler.step() is called in main.py per epoch, not per batch

        # Update learning rate for logging
        for param_group in optimizer.param_groups:
            metric_logger.update(lr=param_group["lr"])

        for k, v in loss_dict.items():
            metric_logger.update(**{f"{k}_loss": v.item()})

        # Save sample images
        if batch_idx % (print_freq * 5) == 0 and hasattr(args, "output_dir"):
            save_sample_images(
                inputs, pred_masks, targets, batch_idx, epoch, args.output_dir
            )

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Train stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args, data_loader, model, epoch, print_freq=100, results_path=None, log_dir="logs"
):
    """Evaluate forgery detection model"""
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    header = f"Test: [{epoch}]"

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            # Handle different batch formats
            if isinstance(batch, dict):
                inputs = batch.get("images", batch.get("inputs")).to(args.device)
                targets = batch.get("masks", batch.get("targets")).to(args.device)
                filenames = batch.get(
                    "filenames",
                    [f"sample_{batch_idx}_{i}" for i in range(inputs.size(0))],
                )
            else:
                # Legacy format
                inputs, targets = batch[0].to(args.device), batch[1].to(args.device)
                filenames = [f"sample_{batch_idx}_{i}" for i in range(inputs.size(0))]

            # Forward pass
            try:
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    pred_masks = outputs.get("mask", outputs.get("output", None))
                else:
                    pred_masks = outputs

                if pred_masks is None:
                    print("Warning: Could not extract predictions from model output")
                    continue

            except Exception as e:
                print(f"Error in model forward pass during evaluation: {e}")
                continue

            # Handle multi-class to binary conversion if needed
            if pred_masks.shape[1] > 1:
                pred_masks_binary = pred_masks.argmax(dim=1, keepdim=True).float()
            else:
                pred_masks_binary = (torch.sigmoid(pred_masks) > 0.5).float()

            # Calculate losses for monitoring
            bce_loss = F.binary_cross_entropy_with_logits(pred_masks, targets.float())
            dice_loss_val = dice_loss(pred_masks, targets)
            focal_loss_val = focal_loss(pred_masks, targets)

            # Update loss metrics
            metric_logger.update(bce_loss=bce_loss.item())
            metric_logger.update(dice_loss=dice_loss_val.item())
            metric_logger.update(focal_loss=focal_loss_val.item())

            # Calculate evaluation metrics
            metrics = compute_metrics(targets, pred_masks_binary)

            for metric_name, metric_value in metrics.items():
                metric_logger.update(**{f"{metric_name}": metric_value})

            # Save evaluation images
            if (
                hasattr(args, "save_images")
                and args.save_images
                and hasattr(args, "output_dir")
            ):
                save_eval_images(
                    inputs, pred_masks, targets, filenames, epoch, args.output_dir
                )

    metric_logger.synchronize_between_processes()
    print(f"Test stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
