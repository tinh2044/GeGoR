import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GCE(nn.Module):
    """
    Geometric-Contrastive Evidence
    """

    def __init__(
        self,
        in_channels: int,
        feat_h: int,
        feat_w: int,
        k_patch: int = 9,
        stride: int = 2,
        m_channels: int = 64,
        emb_dim: int = 128,
        T: int = 16,
        topk: int = 32,
        sigma: float = 3.0,
        alpha: float = 10.0,
        tau: float = 10.0,
        eps: float = 1e-6,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        """
        Args:
            in_channels: C (channels of feature map)
            feat_h, feat_w: h,w of feature map (used to build candidate grid if not supplied)
            k_patch: patch side (k x k)
            stride: stride for candidate grid (s) -> determines K
            m_channels: intermediate channel after 1x1 conv
            emb_dim: embedding dimension d
            T: number of affine hypotheses
            topk: k_nn for sparsify affinity (neighbors kept per row)
            sigma: gating bandwidth for distance gating (in feature pixels)
            alpha: temperature for affinity softmax normalization
            tau: temperature for hypothesis pooling softmax
            eps: numerical eps for L2 normalization
            device: torch.device
        """
        super().__init__()
        self.C = in_channels
        self.h = feat_h
        self.w = feat_w
        self.k = k_patch
        self.s = stride
        self.m = m_channels
        self.d = emb_dim
        self.T = T
        self.topk = topk
        self.sigma = sigma
        self.alpha = alpha
        self.tau = tau
        self.eps = float(eps) if isinstance(eps, str) else eps
        self.device = device

        # Light-weight projector: Conv1x1 (pointwise) then GAP then Linear
        self.pointwise = nn.Conv2d(self.C, self.m, kernel_size=1, bias=True)
        self.proj = nn.Linear(self.m, self.d, bias=True)

        # scoring vector for pooling over hypotheses
        self.r = nn.Parameter(torch.randn(self.d))

        # small initializer for stability
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.r, mean=0.0, std=0.01)

        # Precompute candidate grid Q and mapping M for the configured h,w,s
        Q, M = self._build_candidate_grid_and_mapping(self.h, self.w, self.s)
        # Q: (K,2) coords in (x,y) feature index space, M: (N,K)
        self.register_buffer("Q", Q)  # float
        self.register_buffer("M", M)  # float
        # Also register base P coordinates
        P = self._build_base_grid(self.h, self.w)  # (N,2)
        self.register_buffer("P", P)

    @staticmethod
    def _build_base_grid(h: int, w: int) -> torch.Tensor:
        """Return P coordinates shape (N,2) in (x,y) indexing (x=col, y=row)."""
        ys = torch.arange(0, h, dtype=torch.float32)
        xs = torch.arange(0, w, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N,2)
        return coords  # (N,2)

    @staticmethod
    def _build_candidate_grid_and_mapping(
        h: int, w: int, s: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct candidate grid Q by striding with step s and mapping M (N x K) using bilinear interpolation weights.
        - h,w: feature map spatial size
        - s: stride (s >= 1)
        Returns:
            Q: (K,2) coords in (x,y) (float)
            M: (N,K) mapping (each row sums to 1)
        """
        ys = torch.arange(0, h, step=s, dtype=torch.float32)
        xs = torch.arange(0, w, step=s, dtype=torch.float32)
        qy, qx = torch.meshgrid(ys, xs, indexing="ij")
        qx = qx.reshape(-1)
        qy = qy.reshape(-1)
        Q = torch.stack([qx, qy], dim=-1)  # (K,2)

        # Base grid P
        base_y = torch.arange(0, h, dtype=torch.float32)
        base_x = torch.arange(0, w, dtype=torch.float32)
        by, bx = torch.meshgrid(base_y, base_x, indexing="ij")
        bx = bx.reshape(-1)
        by = by.reshape(-1)
        # N = h*w
        N = h * w
        K = Q.shape[0]

        # For each base coordinate, compute bilinear weights from surrounding candidate nodes
        # normalized coords in candidate-grid index space
        fx = bx / float(s)  # continuous x in candidate grid coords
        fy = by / float(s)
        x0 = torch.floor(fx).long()
        y0 = torch.floor(fy).long()
        x1 = x0 + 1
        y1 = y0 + 1

        # candidate grid dims
        nx = len(xs)
        ny = len(ys)

        # clip
        x0_cl = torch.clamp(x0, 0, nx - 1)
        x1_cl = torch.clamp(x1, 0, nx - 1)
        y0_cl = torch.clamp(y0, 0, ny - 1)
        y1_cl = torch.clamp(y1, 0, ny - 1)

        # fractional parts
        wx = fx - x0.float()
        wy = fy - y0.float()

        # compute indices of the four neighbors in flattened candidate index
        idx00 = (y0_cl * nx + x0_cl).long()  # (N,)
        idx01 = (y0_cl * nx + x1_cl).long()
        idx10 = (y1_cl * nx + x0_cl).long()
        idx11 = (y1_cl * nx + x1_cl).long()

        # weights
        w00 = (1 - wx) * (1 - wy)
        w01 = wx * (1 - wy)
        w10 = (1 - wx) * wy
        w11 = wx * wy

        M = torch.zeros((N, K), dtype=torch.float32)
        M[torch.arange(N), idx00] += w00
        M[torch.arange(N), idx01] += w01
        M[torch.arange(N), idx10] += w10
        M[torch.arange(N), idx11] += w11

        # ensure rows sum to 1 (numerical)
        row_sum = M.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        M = M / row_sum

        return Q, M

    def forward(
        self,
        feature_map: torch.Tensor,
        A_bank: torch.Tensor,
        P: Optional[torch.Tensor] = None,
        Q: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_map: feature map (B, C, h, w)
            A_bank: affine bank, shape either (T, 2, 3) or (T, K, 2, 3)
            P: optional base grid coordinates (N,2). If None, use precomputed P.
            Q: optional candidate coords (K,2). If None, use precomputed Q.
            M: optional mapping (N,K). If None, use precomputed M.

        Returns:
            E: base embeddings (B, N, d)
            A0: affinity matrix (B, N, N) (sparse-like: only topk entries retained per row)
        """
        B, C, h, w = feature_map.shape
        assert C == self.C and h == self.h and w == self.w, (
            "Feature shape mismatch with module config."
        )

        device = feature_map.device
        if P is None:
            P = self.P.to(device)  # (N,2)
        if Q is None:
            Q = self.Q.to(device)  # (K,2)
        if M is None:
            M = self.M.to(device)  # (N,K)

        # unify A_bank to shape (T, 2, 3)
        if A_bank.dim() == 4 and A_bank.shape[1] != 2:
            # fallback: assume shape (T, K, 2, 3)
            if A_bank.dim() == 4:
                # mean over K dimension to obtain per-hypothesis global affine
                # A_bank: (T, K, 2, 3) -> mean -> (T,2,3)
                A_t = A_bank.mean(dim=1)  # (T,2,3)
            else:
                raise ValueError("A_bank shape not understood.")
        elif A_bank.dim() == 3 and A_bank.shape[1] == 2:
            A_t = A_bank  # (T,2,3)
        else:
            A_t = A_bank  # allow direct (T,2,3)

        T = A_t.shape[0]
        assert T == self.T or True  # allow mismatch

        # we'll collect z^{(t)} for each t
        z_list = []

        # Precompute unfold parameters to extract patches at candidate grid:
        # Use F.unfold with kernel_size=k, stride=s, padding=k//2 so that number of patches == K
        pad = self.k // 2
        unfold = nn.Unfold(kernel_size=self.k, dilation=1, padding=pad, stride=self.s)

        # Loop over hypotheses t (T small typically)
        for t_idx in range(T):
            A_single = A_t[t_idx]  # (2,3)
            # If A_single is not normalized for affine_grid, the user must pre-normalize before calling.
            # Expand to batch
            A_batch = A_single.unsqueeze(0).expand(B, -1, -1).to(device)  # (B,2,3)
            # Build grid and warp
            grid = F.affine_grid(
                A_batch, size=(B, C, h, w), align_corners=False
            )  # (B,h,w,2)
            F_warp = F.grid_sample(
                feature_map,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )  # (B,C,h,w)

            # Extract patches at candidate positions via unfold (this yields patches on regular strided grid)
            # patches: (B, C * k * k, K)
            patches = unfold(F_warp)  # (B, C*k*k, K)
            Bp, _, K = patches.shape
            # reshape to (B*K, C, k, k) for conv
            patches = patches.permute(0, 2, 1).contiguous()  # (B, K, C*k*k)
            patches = patches.view(B * K, C, self.k, self.k)  # (B*K, C, k, k)

            # projector: pointwise conv + gap + linear
            x = self.pointwise(patches)  # (B*K, m, k, k)
            # GAP
            x = x.mean(dim=(2, 3))  # (B*K, m)
            z_t = self.proj(x)  # (B*K, d)
            # reshape -> (B, K, d)
            z_t = z_t.view(B, K, self.d)
            # L2 norm
            z_t = z_t / (z_t.norm(dim=-1, keepdim=True).clamp_min(self.eps))
            z_list.append(z_t)

        # stack z: (B, T, K, d)
        Z = torch.stack(z_list, dim=1)  # (B, T, K, d)

        # Hypothesis pooling over T, compute scores s^{(t)} = r^T z^{(t)}
        # r shape (d,) -> expand
        r = self.r.to(device).view(1, 1, 1, self.d)  # (1,1,1,d)
        # compute dot product along last dim
        s = (Z * r).sum(dim=-1)  # (B, T, K)
        # softmax over T
        w = F.softmax(self.tau * s, dim=1)  # (B, T, K)
        w = w.unsqueeze(-1)  # (B, T, K, 1)

        # weighted sum across T: g = sum_t w_t * z_t
        g = (w * Z).sum(dim=1)  # (B, K, d)

        # Lift to base grid P via mapping M (N,K): e = M @ g  -> use einsum
        # M: (N,K) , g: (B,K,d) -> E: (B,N,d)
        E = torch.einsum("nk,bkd->bnd", M.to(device), g)  # (B, N, d)
        # L2 normalize embeddings
        E = E / (E.norm(dim=-1, keepdim=True).clamp_min(self.eps))

        # Build affinity raw via cosine (matrix multiply)
        # E: (B, N, d) -> S = E @ E^T -> (B, N, N)
        S = torch.matmul(E, E.transpose(-1, -2))  # (B, N, N)

        # Distance gating: build G (N,N) once from P coordinates
        # P: (N,2) with coords in feature pixel space (x,y)
        Pcoords = P.to(device)
        px = Pcoords[:, 0].unsqueeze(0)  # (1, N)
        py = Pcoords[:, 1].unsqueeze(0)
        # compute pairwise squared distances (N,N)
        # dx = px - px.T
        diff_x = px.unsqueeze(-1) - px.unsqueeze(-2)  # (1,N,N)
        diff_y = py.unsqueeze(-1) - py.unsqueeze(-2)
        dist2 = diff_x**2 + diff_y**2  # (1,N,N)
        # gating
        G = 1.0 - torch.exp(-dist2 / (2.0 * (self.sigma**2)))  # (1,N,N)
        G = G.squeeze(0)  # (N,N)
        S = S * G.unsqueeze(0)  # (B,N,N) elementwise

        # Sparsify: keep topk per row
        B_, N, _ = S.shape
        k_nn = min(self.topk, N)
        topk_vals, topk_idx = torch.topk(S, k=k_nn, dim=-1)  # (B,N,k)
        S_sparse = torch.zeros_like(S)
        # scatter topk values into zero tensor
        S_sparse = S_sparse.scatter(-1, topk_idx, topk_vals)

        # Two-sided normalization
        L_r = F.softmax(self.alpha * S_sparse, dim=-1)  # row-softmax
        L_c = F.softmax(
            self.alpha * S_sparse, dim=-2
        )  # column-softmax (softmax over rows)
        A0 = L_r * L_c  # elementwise combine -> (B,N,N)

        return {"E": E, "A0": A0}
