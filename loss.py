import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def _to_one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert mask indices (B,H,W) to one-hot (B,C,H,W). If already one-hot, return as is.
    """
    if mask.dim() == 4 and mask.shape[1] == num_classes:
        return mask.float()
    if mask.dim() != 3:
        raise ValueError("mask must be (B,H,W) indices or (B,C,H,W) one-hot")
    b, h, w = mask.shape
    oh = F.one_hot(mask.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    return oh


def _dice_loss_from_logits(
    logits: torch.Tensor, target_oh: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Softmax logits to probabilities and compute mean Dice loss across classes.
    logits: (B,C,H,W), target_oh: (B,C,H,W)
    """
    probs = F.softmax(logits, dim=1)
    num = 2.0 * (probs * target_oh).sum(dim=(2, 3))
    den = probs.pow(2).sum(dim=(2, 3)) + target_oh.pow(2).sum(dim=(2, 3)) + eps
    dice = (num / den).mean(dim=1)  # per-sample mean over classes
    return (1.0 - dice).mean()


def _downsample_binary_map_to_base(
    binary_map: torch.Tensor, base_hw: Tuple[int, int]
) -> torch.Tensor:
    """
    binary_map: (B,1,H,W) float in [0,1]
    returns (B,N) averaged to base grid (h,w) flattened row-major
    """
    b, _, H, W = binary_map.shape
    h, w = base_hw
    ds = F.interpolate(binary_map, size=(h, w), mode="bilinear", align_corners=False)
    return ds.view(b, -1)


def _mine_positive_pairs_from_A2(
    A2: torch.Tensor,
    h: int,
    w: int,
    top_k: int = 5,
    min_dist_px: float = 2.0,
) -> torch.Tensor:
    """
    Mine positive pairs from A2 by taking, for each node i, top_k neighbors j with distance >= min_dist_px.
    Returns indices tensor of shape (B, M, 2) with pairs (i,j) in [0,N).
    """
    b, n, _ = A2.shape
    device = A2.device
    # Coordinates grid
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device).float(),
        torch.arange(w, device=device).float(),
        indexing="ij",
    )
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (N,2)
    d2 = torch.cdist(coords, coords, p=2)  # (N,N)

    # Exclude too-close or self
    mask_far = (d2 >= min_dist_px).float()
    mask_noself = 1.0 - torch.eye(n, device=device)
    valid = mask_far * mask_noself

    pairs_list = []
    for bi in range(b):
        scores = A2[bi] * valid  # (N,N)
        # top_k per row
        topv, topi = torch.topk(scores, k=min(top_k, n - 1), dim=1)
        # Build (i,j) pairs
        ii = torch.arange(n, device=device).unsqueeze(1).expand_as(topi)
        pairs = torch.stack([ii.reshape(-1), topi.reshape(-1)], dim=1)  # (N*top_k,2)
        pairs_list.append(pairs)
    max_m = max(p.shape[0] for p in pairs_list)
    # pad to same length with duplicates (safe for loss averaging)
    out = torch.zeros(b, max_m, 2, dtype=torch.long, device=device)
    for bi, p in enumerate(pairs_list):
        m = p.shape[0]
        out[bi, :m] = p
        if m < max_m:
            out[bi, m:] = p[: (max_m - m)]
    return out  # (B,M,2)


def _info_nce_from_pairs(
    E: torch.Tensor, pos_pairs: torch.Tensor, tau: float = 0.2
) -> torch.Tensor:
    """
    Compute InfoNCE over provided positive pairs.
    E: (B,N,d) L2-normalized
    pos_pairs: (B,M,2) indices (i,j)
    """
    b, n, d = E.shape
    device = E.device
    # Similarity matrices per batch
    sims = torch.bmm(E, E.transpose(1, 2))  # (B,N,N)
    # For numerical stability, subtract max per row if desired (not required for cosine in [-1,1])
    logits = sims / tau
    # Mask out self in denominator by -inf
    eye = torch.eye(n, device=device).unsqueeze(0)
    logits = logits.masked_fill(eye.bool(), float("-inf"))

    losses = []
    for bi in range(b):
        pairs = pos_pairs[bi]  # (M,2)
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        # numerator: exp(logits[i,j])
        num = torch.exp(logits[bi, i_idx, j_idx])
        # denominator: sum_k exp(logits[i,k])
        den = torch.exp(logits[bi, i_idx, :]).sum(dim=1)
        li = -torch.log((num + 1e-12) / (den + 1e-12))
        losses.append(li.mean())
    return torch.stack(losses, dim=0).mean()


class CMFDLoss(nn.Module):
    """
    Five-loss objective for the GCE → GOR pipeline.

    Expected outputs (dict from CMFDNet):
      - 'mask_logits': (B,1,H,W) - binary mask logits
      - 'y_det': (B,1)
      - 'E': (B,N,d)
      - 'A2': (B,N,N)
      - 'A1_map': (B,1,h,w)
      - Optional: 'F_attn' / 'F_attn1', 'F_cooc' / 'F_cooc1' (B,C,h,w) for co-occurrence loss

    Expected targets dict:
      - 'mask': (B,1,H,W) binary mask in [0,1]
      - 'y_det': (B,) or (B,1)
      - Optional: 'pos_pairs': (B,M,2) positive pairs for InfoNCE
    """

    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_det: float = 0.5,
        lambda_gce: float = 0.5,
        lambda_dir: float = 0.3,
        lambda_cooc: float = 0.2,
        dice_weight: float = 0.5,
        nce_top_k: int = 5,
        nce_tau: float = 0.2,
        nce_min_dist_px: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_mask = float(lambda_mask)
        self.lambda_det = float(lambda_det)
        self.lambda_gce = float(lambda_gce)
        self.lambda_dir = float(lambda_dir)
        self.lambda_cooc = float(lambda_cooc)
        self.dice_weight = float(dice_weight)
        self.nce_top_k = int(nce_top_k)
        self.nce_tau = float(nce_tau)
        self.nce_min_dist_px = float(nce_min_dist_px)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=outputs["mask_logits"].device)

        # 1) Mask loss (BCE + Dice)
        mask_logits = outputs["mask_logits"]  # (B,2,H,W) - logits chưa qua softmax
        mask_t = targets.get("mask")
        if mask_t is None:
            raise ValueError("targets['mask'] is required for mask loss")

        # Ensure mask_t has correct shape (B,1,H,W)
        if mask_t.dim() == 3:
            mask_t = mask_t.unsqueeze(1)  # (B,H,W) -> (B,1,H,W)
        elif mask_t.dim() == 4 and mask_t.shape[1] != 1:
            mask_t = mask_t.mean(
                dim=1, keepdim=True
            )  # Convert multi-channel to single channel

        # Convert logits to probabilities using softmax
        mask_probs = F.softmax(mask_logits, dim=1)  # (B,2,H,W)

        # Extract probability of positive class (class 1)
        mask_probs_positive = mask_probs[:, 1:2]  # (B,1,H,W)

        # Binary cross entropy with logits - convert to single channel logits
        # For binary classification with 2-channel logits, use the positive class logit
        mask_logits_single = mask_logits[:, 1:2]  # (B,1,H,W) - logit for positive class
        # Clamp logits to prevent extreme values
        mask_logits_single = torch.clamp(mask_logits_single, -20, 20)
        bce = F.binary_cross_entropy_with_logits(mask_logits_single, mask_t.float())

        # Dice loss for binary mask
        dice_num = 2.0 * (mask_probs_positive * mask_t).sum(dim=(1, 2, 3))
        dice_den = (
            mask_probs_positive.sum(dim=(1, 2, 3)) + mask_t.sum(dim=(1, 2, 3)) + 1e-6
        )
        dice = 1.0 - (dice_num / dice_den).mean()

        L_mask = bce + self.dice_weight * dice
        # Check for NaN and replace with 0 if needed
        if torch.isnan(L_mask) or torch.isinf(L_mask):
            L_mask = torch.tensor(0.0, device=L_mask.device, requires_grad=True)
        loss_dict["L_mask"] = L_mask
        total = total + self.lambda_mask * L_mask

        # 2) Detection loss (BCE)
        y_det_hat = outputs["y_det"].view(-1)
        y_det = targets.get("y_det")
        if y_det is not None:
            y_det = y_det.view(-1).float()
            # Use binary cross entropy on probabilities (y_det_hat already sigmoid in net). Use small clamp for stability.
            y_det_hat = torch.clamp(y_det_hat, 1e-6, 1 - 1e-6)
            L_det = F.binary_cross_entropy(y_det_hat, y_det)
            # Check for NaN and replace with 0 if needed
            if torch.isnan(L_det) or torch.isinf(L_det):
                L_det = torch.tensor(0.0, device=L_det.device, requires_grad=True)
            loss_dict["L_det"] = L_det
            total = total + self.lambda_det * L_det

        # Base grid size for node-level losses
        A1_map = outputs.get("A1_map")
        if A1_map is None:
            raise ValueError("outputs['A1_map'] required to infer base grid size")
        _, _, h, w = A1_map.shape

        # 3) GCE InfoNCE loss on E, mined from A2 if no pairs given
        E = outputs.get("E")  # (B,N,d)
        A2 = outputs.get("A2")  # (B,N,N)
        if E is not None and A2 is not None:
            pos_pairs = targets.get("pos_pairs")
            if pos_pairs is None:
                pos_pairs = _mine_positive_pairs_from_A2(
                    A2, h=h, w=w, top_k=self.nce_top_k, min_dist_px=self.nce_min_dist_px
                )
            L_gce = _info_nce_from_pairs(E, pos_pairs, tau=self.nce_tau)
            # Check for NaN and replace with 0 if needed
            if torch.isnan(L_gce) or torch.isinf(L_gce):
                L_gce = torch.tensor(0.0, device=L_gce.device, requires_grad=True)
            loss_dict["L_gce"] = L_gce
            total = total + self.lambda_gce * L_gce

        # 4) Directionality loss on node priors (pi_src, pi_tgt)
        pi_src = outputs.get("pi_src")
        pi_tgt = outputs.get("pi_tgt")
        if pi_src is not None and pi_tgt is not None:
            # For binary mask, treat the entire mask as both source and target
            # This is a simplified approach for binary forgery detection
            mask_binary = mask_t.float()  # (B,1,H,W)
            mask_ds = _downsample_binary_map_to_base(mask_binary, (h, w))  # (B,N)

            # BCE on probabilities (pi_* already in [0,1])
            pi_src = pi_src.view(mask_ds.shape).clamp(1e-6, 1 - 1e-6)
            pi_tgt = pi_tgt.view(mask_ds.shape).clamp(1e-6, 1 - 1e-6)

            # Both pi_src and pi_tgt should match the binary mask
            L_dir = F.binary_cross_entropy(pi_src, mask_ds) + F.binary_cross_entropy(
                pi_tgt, mask_ds
            )
            # Check for NaN and replace with 0 if needed
            if torch.isnan(L_dir) or torch.isinf(L_dir):
                L_dir = torch.tensor(0.0, device=L_dir.device, requires_grad=True)
            loss_dict["L_dir"] = L_dir
            total = total + self.lambda_dir * L_dir

        # 5) Co-occurrence consistency loss (optional if tensors provided)
        F_attn = outputs.get("F_attn")
        F_cooc = outputs.get("F_cooc")
        if F_attn is None:
            # try branch-1 naming
            F_attn = outputs.get("F_attn1")
        if F_cooc is None:
            F_cooc = outputs.get("F_cooc1")
        if A2 is not None and F_attn is not None and F_cooc is not None:
            b, c, hh, ww = F_attn.shape
            assert (hh, ww) == (h, w), "F_attn must be at base grid size (h,w)"
            X = F_attn.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
            Y = torch.bmm(A2, X)  # (B,N,C)
            Y_map = Y.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            L_cooc = F.l1_loss(Y_map, F_cooc)
            # Check for NaN and replace with 0 if needed
            if torch.isnan(L_cooc) or torch.isinf(L_cooc):
                L_cooc = torch.tensor(0.0, device=L_cooc.device, requires_grad=True)
            loss_dict["L_cooc"] = L_cooc
            total = total + self.lambda_cooc * L_cooc

        # Check total loss for NaN/Inf and clamp if needed
        if torch.isnan(total) or torch.isinf(total):
            total = torch.tensor(0.0, device=total.device, requires_grad=True)
        loss_dict["total"] = total
        return loss_dict


__all__ = ["CMFDLoss"]
