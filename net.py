import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from gce import GCE
from gor import GOR
from loss import CMFDLoss


class ExtractorEfficientNet(nn.Module):
    def __init__(
        self,
        variant: str = "b0",
        pretrained: bool = True,
        out_channels: int = 256,
        r_target: int = 8,
    ):
        super().__init__()
        assert r_target >= 1 and int(r_target) == r_target
        self.r_target = int(r_target)
        self.out_ch = out_channels

        try:
            weights = getattr(
                models, f"EfficientNet_{variant.upper()}_Weights"
            ).IMAGENET1K_V1
            base = getattr(models, f"efficientnet_{variant}")(weights=weights).features

        except AttributeError:
            raise ValueError(f"Variant {variant} not found in torchvision.models")

        self.backbone_blocks = nn.ModuleList([b for b in base])
        self.proj = nn.Conv2d(
            self._infer_backbone_out_channels(), out_channels, kernel_size=1
        )

        nn.init.kaiming_normal_(
            self.proj.weight, a=0, mode="fan_out", nonlinearity="relu"
        )
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def _infer_backbone_out_channels(self):
        try:
            last = self.backbone_blocks[-1]
            for module in reversed(list(last.modules())):
                if isinstance(module, nn.Conv2d):
                    return module.out_channels
        except Exception:
            pass
        return 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C0, H0, W0 = x.shape
        target_h = max(1, H0 // self.r_target)
        target_w = max(1, W0 // self.r_target)

        cur = x
        for block in self.backbone_blocks:
            cur = block(cur)

        out = self.proj(cur)

        if out.shape[-2] != target_h or out.shape[-1] != target_w:
            out = F.interpolate(
                out, size=(target_h, target_w), mode="bilinear", align_corners=True
            )

        return out


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r)
                for r in rates
            ]
        )
        self.fuse = nn.Conv2d(len(rates) * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.fuse(x)


class A1Refine(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, t_map: torch.Tensor) -> torch.Tensor:
        return self.net(t_map)


class ConvFuse(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 256, out_ch: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DetectionHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.mlp(z)


class LocalizationHead(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 128, out_ch: int = 3) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(mid_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.out_conv(x)
        Ht, Wt = target_hw
        if x.shape[-2:] != (Ht, Wt):
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=True)
        return x


def _make_base_centers(h: int, w: int, device: torch.device) -> torch.Tensor:
    xs = torch.arange(0, w, device=device, dtype=torch.float32)
    ys = torch.arange(0, h, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (N,2)


def _identity_affine_bank(
    T: int, N: int, noise: float = 0.0, device=None
) -> torch.Tensor:
    A = torch.zeros(T, N, 2, 3, device=device)
    A[:, :, 0, 0] = 1.0
    A[:, :, 1, 1] = 1.0
    if noise > 0.0:
        A = A + noise * torch.randn_like(A)
    return A


class CMFDNet(nn.Module):
    def __init__(
        self,
        encoder_out_ch: int = 256,
        gce_embed_dim: int = 256,
        gce_patch_size: int = 9,
        aspp_out_ch: int = 128,
        fuse_out_ch: int = 256,
        T_affine: int = 4,
        num_cls: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = ExtractorEfficientNet(
            pretrained=True,
            out_channels=encoder_out_ch,
            variant=kwargs.get("encoder_variant", "b0"),
        )
        feat_h = kwargs.get("feat_h", 256 // 8)
        feat_w = kwargs.get("feat_w", 256 // 8)

        gce_kwargs = kwargs.get("gce", {}).copy()
        if "eps" not in gce_kwargs:
            gce_kwargs["eps"] = 1e-6

        self.gce = GCE(
            feat_h=feat_h,
            feat_w=feat_w,
            in_channels=encoder_out_ch,
            emb_dim=gce_embed_dim,
            k_patch=gce_patch_size,
            **gce_kwargs,
        )

        gor_kwargs = kwargs.get("gor", {}).copy()
        gor_kwargs["feat_h"] = feat_h
        gor_kwargs["feat_w"] = feat_w

        self.gor = GOR(**gor_kwargs)
        self.aspp1 = ASPP(encoder_out_ch, aspp_out_ch)
        self.aspp2 = ASPP(encoder_out_ch, aspp_out_ch)
        self.a1_refine = A1Refine(hidden=32)
        fuse_in_ch = 4 * aspp_out_ch + 4
        self.fuse = ConvFuse(in_ch=fuse_in_ch, hidden=256, out_ch=fuse_out_ch)
        self.det_head = DetectionHead(in_ch=fuse_out_ch, hidden=128)
        self.loc_head = LocalizationHead(in_ch=fuse_out_ch, mid_ch=128, out_ch=num_cls)
        self.T_affine = int(T_affine)

        self.loss_fn = CMFDLoss(**kwargs.get("loss", {}))

    def forward(self, x: torch.Tensor, gt_mask: torch.Tensor = None) -> dict:
        B, _, H, W = x.shape
        device = x.device
        Fmap = self.backbone(x)  # (B,C,h,w)
        _, C, h, w = Fmap.shape
        N = h * w

        P = _make_base_centers(h, w, device)  # (N,2)
        A_bank = _identity_affine_bank(
            self.T_affine, N, noise=0.0, device=device
        )  # (T,N,2,3)

        gce_out = self.gce(Fmap, A_bank, M=None, P=P)
        E = gce_out["E"]  # (B,N,d) unused here but available
        A0 = gce_out["A0"]  # (B,N,N)

        gor_out = self.gor(E, A0, P)
        A2 = gor_out["A2"]  # (B,N,N)
        U = gor_out["U"]  # (B,N)
        pi_src = gor_out["pi_src"]  # (B,N)
        pi_tgt = gor_out["pi_tgt"]  # (B,N)
        A1_map = gor_out["A1_map"]  # (B,1,h,w)
        A1_map = self.a1_refine(A1_map)  # (B,1,h,w)

        F_aspp1 = self.aspp1(Fmap)
        F_aspp2 = self.aspp2(Fmap)

        F_attn1 = F_aspp1 * A1_map
        F_attn2 = F_aspp2 * A1_map

        def propagate(F_attn: torch.Tensor, A2_op: torch.Tensor) -> torch.Tensor:
            B_, C_, H_, W_ = F_attn.shape
            X = F_attn.permute(0, 2, 3, 1).contiguous().view(B_, H_ * W_, C_)
            Y = torch.bmm(A2_op, X)  # (B,N,C)
            return Y.view(B_, H_, W_, C_).permute(0, 3, 1, 2).contiguous()

        F_cooc1 = propagate(F_attn1, A2)
        F_cooc2 = propagate(F_attn2, A2)

        def vec_to_map(v: torch.Tensor, hh: int, ww: int) -> torch.Tensor:
            return v.view(v.shape[0], 1, hh, ww)

        U_map = vec_to_map(U, h, w)
        pi_src_map = vec_to_map(pi_src, h, w)
        pi_tgt_map = vec_to_map(pi_tgt, h, w)

        fused = torch.cat(
            [F_attn1, F_attn2, F_cooc1, F_cooc2, A1_map, U_map, pi_src_map, pi_tgt_map],
            dim=1,
        )
        F_final = self.fuse(fused)

        y_det = self.det_head(F_final)
        mask_logits = self.loc_head(F_final, target_hw=(H, W))
        mask_probs = F.softmax(mask_logits, dim=1)
        mask = mask_probs[:, 1:2]

        loss_dict = None
        if gt_mask is not None:
            mask_sum = torch.sum(gt_mask, dim=[1, 2, 3])
            is_tampered = (mask_sum > 0).float()  # (B,) tensor

            loss_dict = self.loss_fn(
                outputs={
                    "y_det": y_det,
                    "mask_logits": mask_logits,
                    "A1_map": A1_map,
                    "E": E,
                    "A2": A2,
                    "pi_src": pi_src_map,
                    "pi_tgt": pi_tgt_map,
                },
                targets={"y_det": is_tampered, "mask": gt_mask},
            )

        return {
            "features_base": Fmap,
            "E": E,
            "A0": A0,
            "A2": A2,
            "A1_map": A1_map,
            "U": U_map,
            "pi_src": pi_src_map,
            "pi_tgt": pi_tgt_map,
            "F_final": F_final,
            "y_det": y_det,
            "mask_logits": mask_logits,
            "mask": mask,
            "loss": loss_dict,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CMFDNet()
    num_parameter = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameter}")

    x = torch.randn(1, 3, 256, 256)
    gt_mask = torch.randint(0, 2, (1, 1, 256, 256)).float()

    print(f"Input shape: {x.shape}")
    print(f"GT mask shape: {gt_mask.shape}")
    print(f"GT mask unique values: {torch.unique(gt_mask)}")

    print("Testing forward without loss...")
    out_no_loss = model(x)
    print(f"Forward without loss successful. Output keys: {list(out_no_loss.keys())}")

    print("Testing forward with loss...")
    out = model(x, gt_mask=gt_mask)

    print("Test Loss Calculation:")
    print(f"Total Loss: {out['loss']['total']:.6f}")
    print(f"Model Output Keys: {list(out.keys())}")
    print(
        f"Output Shapes: { {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)} }"
    )

    print(f"Mask Loss: {out['loss']['total'].item():.6f}")
    print("Forward pass completed successfully!")

    for k, v in out["loss"].items():
        print(f"{k}: {v.item():.6f}")
