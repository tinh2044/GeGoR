import torch
from typing import Dict


def calculate_iou(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate Intersection over Union (IoU) for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        IoU score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection

    return (intersection / (union + 1e-8)).item()


def calculate_f1(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate F1 score for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        F1 score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    tp = ((pred_binary == 1) & (gt_binary == 1)).sum().float()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum().float()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()


def calculate_precision(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate precision for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        Precision score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    tp = ((pred_binary == 1) & (gt_binary == 1)).sum().float()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum().float()

    return (tp / (tp + fp + 1e-8)).item()


def calculate_recall(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate recall for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        Recall score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    tp = ((pred_binary == 1) & (gt_binary == 1)).sum().float()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum().float()

    return (tp / (tp + fn + 1e-8)).item()


def calculate_accuracy(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Calculate pixel accuracy for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        Pixel accuracy score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    correct = (pred_binary == gt_binary).sum().float()
    total = torch.numel(gt_binary)

    return (correct / total).item()


def compute_metrics(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all forgery detection metrics
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        dict with keys: 'iou', 'f1', 'precision', 'recall', 'accuracy'
    """
    return {
        "iou": calculate_iou(pred_mask, gt_mask, threshold),
        "f1": calculate_f1(pred_mask, gt_mask, threshold),
        "precision": calculate_precision(pred_mask, gt_mask, threshold),
        "recall": calculate_recall(pred_mask, gt_mask, threshold),
        "accuracy": calculate_accuracy(pred_mask, gt_mask, threshold),
    }
