import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GOR(nn.Module):
    """
    Graphical Offset Reasoner
    """

    def __init__(
        self,
        feat_h: int,
        feat_w: int,
        topk: int = 32,
        K_o: int = 32,
        tau: float = 1.5,
        eta: float = 8.0,
        lambda0: float = 1e-3,
        alpha: float = 10.0,
        a: float = 1.0,
        b: float = 0.0,
        beta: float = 8.0,
        beta_prime: float = 8.0,
        smooth_ks: int = 5,
        smooth_sigma: float = 1.0,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.h = feat_h
        self.w = feat_w
        self.N = feat_h * feat_w
        self.topk = min(topk, self.N)
        self.K_o = K_o
        self.tau = tau
        self.eta = eta
        self.lambda0 = lambda0
        self.alpha = alpha
        self.a = a
        self.b = b
        self.beta = beta
        self.beta_prime = beta_prime
        self.smooth_ks = smooth_ks
        self.smooth_sigma = smooth_sigma
        self.eps = float(eps) if isinstance(eps, str) else eps
        self.device = device if device is not None else torch.device("cpu")

        # Prebuild base coordinates P (N,2) row-major (x, y)
        P = self._build_base_grid(self.h, self.w)  # (N,2)
        self.register_buffer("P", P)  # float

        # Prebuild offset bins (K_o,2). Use polar uniformly distributed bins (r scales + angles)
        bins = self._build_offset_bins(self.K_o, self.h, self.w)  # (K_o, 2)
        self.register_buffer("bins", bins)

        # Prebuild Gaussian smoothing kernel for A1 map (1x1 conv kernel)
        kernel = self._gaussian_kernel(self.smooth_ks, self.smooth_sigma)
        self.register_buffer("smooth_kernel", kernel)  # (1,1,ks,ks)

    @staticmethod
    def _build_base_grid(h: int, w: int) -> torch.Tensor:
        ys = torch.arange(0, h, dtype=torch.float32)
        xs = torch.arange(0, w, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N,2) as (x,y)
        return coords

    @staticmethod
    def _build_offset_bins(K_o: int, h: int, w: int) -> torch.Tensor:
        """
        Build K_o bins distributed in polar coordinates up to a radius.
        Returns (K_o, 2) as (dx, dy) in feature-space units.
        """
        # radius up to half-diagonal of feature grid
        maxr = ((h**2 + w**2) ** 0.5) * 0.5
        # Use rings: choose number of rings and angles per ring to approximate K_o points
        # We'll generate K_o points on circle with radii spreading from small -> large
        ks = torch.arange(1, K_o + 1, dtype=torch.float32)
        angles = 2.0 * torch.pi * (ks - 1) / K_o
        # radii: small ramp (0.2..0.8) times maxr
        radii = 0.2 * maxr + 0.6 * maxr * (ks - 1) / (K_o - 1 + 1e-9)
        bx = radii * torch.cos(angles)
        by = radii * torch.sin(angles)
        bins = torch.stack([bx, by], dim=-1)  # (K_o,2)
        return bins

    @staticmethod
    def _gaussian_kernel(ks: int, sigma: float) -> torch.Tensor:
        """Return kernel shape (1,1,ks,ks) normalized sum=1"""
        ax = torch.arange(-ks // 2 + 1, ks // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, ks, ks).float()

    def forward(
        self, E: torch.Tensor, A0: torch.Tensor, P: Optional[torch.Tensor] = None
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Args:
            E: (B, N, d)
            A0: (B, N, N)
            P: optional (N,2)
        Returns:
            A2_dense: (B, N, N) row-normalized refined affinity
            U: (B, N) backgroundness in (0,1)
            pi_src: (B, N)
            pi_tgt: (B, N)
            A1_map: (B, h, w) attention spatial map
            A2_prop: same as A2_dense (operator for propagation)
        """
        device = E.device
        B = E.shape[0]
        N = self.N
        assert A0.shape[0] == B and A0.shape[1] == N and A0.shape[2] == N, (
            "A0 shape mismatch"
        )

        if P is None:
            Pcoords = self.P.to(device)  # (N,2)
        else:
            Pcoords = P.to(device)

        bins = self.bins.to(device)  # (K_o,2)

        k_nn = min(self.topk, N)
        # topk values and indices per row
        topk_vals, topk_idx = torch.topk(
            A0, k=k_nn, dim=-1
        )  # shapes (B,N,k_nn), (B,N,k_nn)

        p_j = Pcoords[topk_idx.view(-1)].view(B, N, k_nn, 2)  # (B,N,k,2)
        # p_i: per row's base coordinate
        p_i = (
            Pcoords.unsqueeze(0).unsqueeze(2).expand(B, N, k_nn, 2).to(device)
        )  # (B,N,k,2)

        # offset vectors
        v = p_j - p_i  # (B,N,k,2)

        v_exp = v.unsqueeze(-2)  # (B,N,k,1,2)
        bins_exp = bins.view(1, 1, 1, self.K_o, 2)  # (1,1,1,K_o,2)
        diff = v_exp - bins_exp  # (B,N,k,K_o,2)
        dist2 = (diff**2).sum(dim=-1)  # (B,N,k,K_o)
        kappa = torch.exp(-dist2 / (2.0 * (self.tau**2) + 1e-12))  # (B,N,k,K_o)

        S_vals = topk_vals  # (B,N,k)
        # multiply and sum over i and neighbor -> H: (B, K_o)
        H = (S_vals.unsqueeze(-1) * kappa).sum(
            dim=(1, 2)
        )  # sum over i and neighbors -> (B, K_o)

        W = F.softmax(self.eta * H, dim=-1)  # (B, K_o)

        W_exp = W.view(B, 1, 1, self.K_o)  # (B,1,1,K_o)
        R_edges = self.lambda0 + (W_exp * kappa).sum(dim=-1)  # (B, N, k)

        tildeA_edges = S_vals * R_edges  # (B,N,k)
        row_sums = tildeA_edges.sum(dim=-1, keepdim=True)  # (B,N,1)
        row_sums = row_sums + self.eps
        A1_edges = tildeA_edges / row_sums  # (B,N,k)

        # ----------------------------
        # 7) Entropy per row (over neighbors): p = softmax(alpha * A1_edges) then entropy
        # ----------------------------
        p = F.softmax(self.alpha * A1_edges, dim=-1)  # (B,N,k)
        entropy = -(p * (torch.log(p + self.eps))).sum(dim=-1)  # (B,N)
        # Backgroundness U via sigmoid mapping: U = sigmoid(a * (-H_row) + b)
        U = torch.sigmoid(self.a * (-entropy) + self.b)  # (B,N)
        m = 1.0 - U  # retention coefficient (B,N)

        m_i = m.unsqueeze(-1)  # (B,N,1)
        idx_flat = topk_idx  # (B,N,k)
        m_j = torch.gather(m.unsqueeze(1).expand(-1, N, -1), 2, idx_flat)  # (B,N,k)
        # compute hatA
        hatA_edges = m_i * A1_edges * m_j  # (B,N,k)

        # row-normalize to get A2 edges
        row_sum2 = hatA_edges.sum(dim=-1, keepdim=True) + self.eps
        A2_edges = hatA_edges / row_sum2  # (B,N,k)

        A2_dense = torch.zeros((B, N, N), device=device, dtype=A2_edges.dtype)
        for b in range(B):
            A2_dense[b].scatter_(1, topk_idx[b], A2_edges[b])

        # Ensure rows sum approx 1 (numerical)
        row_sum_dense = A2_dense.sum(dim=-1, keepdim=True) + self.eps
        A2_dense = A2_dense / row_sum_dense

        exp_fwd = torch.exp(self.beta * A2_dense)
        exp_rev = torch.exp(self.beta * A2_dense.transpose(-1, -2))
        denom = exp_fwd + exp_rev + self.eps
        P_dir = exp_fwd / denom  # (B,N,N)
        # zero diagonal
        diag_idx = torch.arange(N, device=device)
        P_dir[:, diag_idx, diag_idx] = 0.0

        s_evidence = (A2_dense * P_dir).sum(dim=-1)  # (B,N)
        # compute t: A2_dense_transpose * (1 - P_dir_transpose) summed over rows => equivalently:
        t_evidence = (A2_dense.transpose(-1, -2) * (1.0 - P_dir.transpose(-1, -2))).sum(
            dim=-1
        )  # (B,N)
        # normalize to priors
        denom_st = s_evidence + t_evidence + self.eps
        pi_src = s_evidence / denom_st  # (B,N)
        pi_tgt = 1.0 - pi_src

        W_mean = W.mean(dim=0)  # (K_o,)
        Wm_exp = W_mean.view(1, 1, 1, self.K_o)  # (1,1,1,K_o)
        # kappa currently is (B,N,k,K_o)
        # compute chi_edges per batch: (B,N,k) = sum_k W_mean[k] * kappa_{b,i,j,k}
        chi_edges = (kappa * Wm_exp.to(device)).sum(dim=-1)  # (B,N,k)

        # local smoothing weights w_{b,i,j} = softmax(beta' * A2_edges) along neighbors
        w_local = F.softmax(self.beta_prime * A2_edges, dim=-1)  # (B,N,k)
        # node-level intensity q
        q_nodes = (w_local * chi_edges).sum(dim=-1)  # (B,N)

        A1_map = q_nodes.view(B, self.h, self.w).unsqueeze(1)  # (B,1,h,w)
        A1_smoothed = F.conv2d(
            A1_map, self.smooth_kernel.to(device), padding=self.smooth_ks // 2
        )
        A1_map = torch.sigmoid(A1_smoothed)  # (B, 1, h, w)

        # A2_prop operator (same as A2_dense row-normalized)
        A2_prop = A2_dense

        return {
            "A2": A2_dense,
            "U": U,
            "pi_src": pi_src,
            "pi_tgt": pi_tgt,
            "A1_map": A1_map,
            "A2_prop": A2_prop,
        }
