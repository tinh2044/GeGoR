import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
import yaml
import random
from pathlib import Path
from thop import profile, clever_format

from net import (
    CMFDNet,
    ExtractorEfficientNet,
    GCE,
    GOR,
    ASPP,
    A1Refine,
    ConvFuse,
    DetectionHead,
    LocalizationHead,
    _make_base_centers,
    _identity_affine_bank,
)
import utils


def get_args_parser():
    parser = argparse.ArgumentParser("Calculator FLOPs and Parameters", add_help=False)
    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/casia2.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Image size for FLOPs calculation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for FLOPs calculation"
    )

    return parser


def count_module_flops(model, input_tensor, module_name):
    """Calculate FLOPs for a specific module"""
    try:
        # Handle different input types (single tensor or tuple)
        if isinstance(input_tensor, tuple):
            macs, params = profile(model, inputs=input_tensor, verbose=False)
        else:
            macs, params = profile(model, inputs=(input_tensor,), verbose=False)

        flops = 2 * macs  # Convert MACs to FLOPs (multiply-accumulate operations)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return {
            "flops": flops,
            "params": params,
            "flops_str": flops_str,
            "params_str": params_str,
            "macs": macs,
        }
    except Exception as e:
        print(f"  ⚠️  Error calculating FLOPs for {module_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def analyze_module_details(model, input_shape, device):
    """Analyze each module in detail"""
    print("=" * 100)
    print("🔍 DETAILED MODULE ANALYSIS")
    print("=" * 100)

    batch_size = input_shape[0]
    module_results = {}

    print(f"\n🔍 Input shape for detailed analysis: {input_shape}")

    # 1. ExtractorEfficientNet
    print("\n📊 1. ExtractorEfficientNet (Feature Extractor)")
    print("-" * 50)

    extractor = ExtractorEfficientNet(pretrained=False).to(device)
    x = torch.randn(input_shape).to(device)
    extractor_result = count_module_flops(extractor, x, "ExtractorEfficientNet")

    if extractor_result:
        print(f"   • Input shape: {x.shape}")
        output = extractor(x)
        print(f"   • Output shape: {output.shape}")
        print(f"   • Parameters: {extractor_result['params_str']}")
        print(f"   • FLOPs: {extractor_result['flops_str']}")
        module_results["ExtractorEfficientNet"] = extractor_result

    # 2. GCE
    print("\n📊 2. Geometric-Contrastive Evidence (GCE)")
    print("-" * 50)

    # Get feature map from extractor
    with torch.no_grad():
        fmap = extractor(x)  # (B, C, H/8, W/8)

    B, C, h, w = fmap.shape
    N = h * w

    # Create required inputs for GCE using functions from net.py
    P = _make_base_centers(h, w, device)  # (N, 2)

    # A_bank should be (T, N, 2, 3) - affine transformations
    T_affine = 4
    A_bank = _identity_affine_bank(T_affine, N, noise=0.0, device=device)

    gce_module = GCE(
        in_channels=C,
        feat_h=h,
        feat_w=w,
        emb_dim=256,
        k_patch=9,
        eps=1e-6,
        device=device,
    ).to(device)

    try:
        # Test forward first to ensure it works
        gce_out = gce_module(fmap, A_bank, M=None, P=P)
        print(
            f"   • GCE forward successful - A0: {gce_out['A0'].shape}, E: {gce_out['E'].shape}"
        )

        # Now calculate FLOPs
        gce_result = count_module_flops(gce_module, (fmap, A_bank, None, P), "GCE")
        if gce_result:
            print(f"   • Feature input: {fmap.shape}")
            print(f"   • Coordinates P: {P.shape}")
            print(f"   • Affine bank: {A_bank.shape}")
            print(f"   • Parameters: {gce_result['params_str']}")
            print(f"   • FLOPs: {gce_result['flops_str']}")
            module_results["GCE"] = gce_result
    except Exception as e:
        print(f"   ⚠️  Error with GCE: {e}")
        import traceback

        traceback.print_exc()

    # 3. GOR
    print("\n📊 3. Graphical Offset Reasoner (GOR)")
    print("-" * 50)

    try:
        # Mock A0 for GOR input (from GCE output)
        A0 = torch.randn(B, N, N).to(device)
        A0 = F.softmax(A0, dim=-1)  # Normalize like affinity matrix

        # Create mock E for GOR input
        E = torch.randn(B, N, 256).to(device)

        gor_module = GOR(feat_h=h, feat_w=w, device=device).to(device)

        # Test forward first
        gor_out = gor_module(E, A0, P)
        print(f"   • GOR forward successful - A1_map: {gor_out['A1_map'].shape}")

        # Now calculate FLOPs
        gor_result = count_module_flops(gor_module, (E, A0, P), "GOR")
        if gor_result:
            print(f"   • Affinity input: {A0.shape}")
            print(f"   • Coordinates P: {P.shape}")
            print(f"   • Grid size: {h}x{w}")
            print(f"   • pi_src: {gor_out['pi_src'].shape}")
            print(f"   • pi_tgt: {gor_out['pi_tgt'].shape}")
            print(f"   • Parameters: {gor_result['params_str']}")
            print(f"   • FLOPs: {gor_result['flops_str']}")
            module_results["GOR"] = gor_result
    except Exception as e:
        print(f"   ⚠️  Error with GOR: {e}")
        import traceback

        traceback.print_exc()

    # 4. ASPP
    print("\n📊 4. Atrous Spatial Pyramid Pooling (ASPP)")
    print("-" * 50)

    aspp_module = ASPP(in_ch=C, out_ch=128).to(device)
    aspp_result = count_module_flops(aspp_module, fmap, "ASPP")

    if aspp_result:
        aspp_out = aspp_module(fmap)
        print(f"   • Input: {fmap.shape}")
        print(f"   • Output: {aspp_out.shape}")
        print(f"   • Parameters: {aspp_result['params_str']}")
        print(f"   • FLOPs: {aspp_result['flops_str']}")
        module_results["ASPP"] = aspp_result

    # 5. A1Refine
    print("\n📊 5. A1 Attention Refinement")
    print("-" * 50)

    a1_refine = A1Refine(hidden=32).to(device)
    a1_input = torch.randn(B, 1, h, w).to(device)
    a1_result = count_module_flops(a1_refine, a1_input, "A1Refine")

    if a1_result:
        a1_out = a1_refine(a1_input)
        print(f"   • Input: {a1_input.shape}")
        print(f"   • Output: {a1_out.shape}")
        print(f"   • Parameters: {a1_result['params_str']}")
        print(f"   • FLOPs: {a1_result['flops_str']}")
        module_results["A1Refine"] = a1_result

    # 6. ConvFuse
    print("\n📊 6. Convolution Fusion (ConvFuse)")
    print("-" * 50)

    # Mock fusion input (4 ASPP channels + 4 single channels)
    fuse_in_ch = 4 * 128 + 4
    conv_fuse = ConvFuse(in_ch=fuse_in_ch, hidden=256, out_ch=256).to(device)
    fuse_input = torch.randn(B, fuse_in_ch, h, w).to(device)
    fuse_result = count_module_flops(conv_fuse, fuse_input, "ConvFuse")

    if fuse_result:
        fuse_out = conv_fuse(fuse_input)
        print(f"   • Input: {fuse_input.shape}")
        print(f"   • Output: {fuse_out.shape}")
        print(f"   • Parameters: {fuse_result['params_str']}")
        print(f"   • FLOPs: {fuse_result['flops_str']}")
        module_results["ConvFuse"] = fuse_result

    # 7. DetectionHead
    print("\n📊 7. Detection Head")
    print("-" * 50)

    det_head = DetectionHead(in_ch=256, hidden=128).to(device)
    det_result = count_module_flops(det_head, fuse_out, "DetectionHead")

    if det_result:
        det_out = det_head(fuse_out)
        print(f"   • Input: {fuse_out.shape}")
        print(f"   • Output: {det_out.shape}")
        print(f"   • Parameters: {det_result['params_str']}")
        print(f"   • FLOPs: {det_result['flops_str']}")
        module_results["DetectionHead"] = det_result

    # 8. LocalizationHead
    print("\n📊 8. Localization Head")
    print("-" * 50)

    H, W = input_shape[2], input_shape[3]  # Original image size
    loc_head = LocalizationHead(in_ch=256, mid_ch=128, out_ch=2).to(device)

    try:
        # Test forward first
        loc_out = loc_head(fuse_out, target_hw=(H, W))
        print(f"   • LocalizationHead forward successful - Output: {loc_out.shape}")

        # Now calculate FLOPs - use tuple for target_hw parameter
        loc_result = count_module_flops(
            loc_head, (fuse_out, (H, W)), "LocalizationHead"
        )
        if loc_result:
            print(f"   • Input: {fuse_out.shape}")
            print(f"   • Target size: {H}x{W}")
            print(f"   • Parameters: {loc_result['params_str']}")
            print(f"   • FLOPs: {loc_result['flops_str']}")
            module_results["LocalizationHead"] = loc_result
    except Exception as e:
        print(f"   ⚠️  Error with LocalizationHead: {e}")
        import traceback

        traceback.print_exc()

    return module_results


def main(args, cfg):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg_data = cfg.get("data", {})
    cfg_model = cfg.get("model", {})

    # Override image size if provided (command line takes priority)
    image_size = args.image_size
    if image_size == 256 and "image_size" in cfg_data:
        # Only use config image_size if command line is default (256)
        image_size = cfg_data.get("image_size", image_size)
    print(
        f"Using image size: {image_size} (command line: {args.image_size}, config: {cfg_data.get('image_size', 'not set')})"
    )

    batch_size = args.batch_size
    input_shape = (batch_size, 3, image_size, image_size)

    print(f"Input shape for FLOPs calculation: {input_shape}")

    try:
        # Calculate expected feature dimensions from extractor
        test_extractor = ExtractorEfficientNet().to(device)
        with torch.no_grad():
            test_input = torch.randn(1, 3, image_size, image_size).to(device)
            test_output = test_extractor(test_input)
            _, _, feat_h, feat_w = test_output.shape

        cfg_model_updated = cfg_model.copy()
        cfg_model_updated.update({"feat_h": feat_h, "feat_w": feat_w})

        model = CMFDNet(**cfg_model_updated)
        model = model.to(device)
        n_parameters = utils.count_model_parameters(model)
        print(f"Model created successfully with {n_parameters:,} parameters")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Model config:", cfg_model)
        raise

    # Analyze detailed modules
    module_results = analyze_module_details(model, input_shape, device)

    # Summary
    print("\n" + "=" * 100)
    print("📊 SUMMARY - MODULE FLOPs BREAKDOWN")
    print("=" * 100)

    total_flops = 0
    total_params = 0

    for module_name, result in module_results.items():
        if result:
            flops_val = result["flops"]
            params_val = result["params"]
            total_flops += flops_val
            total_params += params_val
    for module_name, result in module_results.items():
        if result:
            flops_val = result["flops"]
            params_val = result["params"]
            print(
                f"{module_name:15}: {result['flops_str']:>10} FLOPs({result['flops'] / total_flops:.2f}), {result['params_str']:>10} params({result['params'] / total_params:.2f})"
            )

    print("-" * 80)
    total_flops_str, total_params_str = clever_format(
        [total_flops, total_params], "%.3f"
    )
    print(f"{'TOTAL':15}: {total_flops_str:>10} FLOPs, {total_params_str:>10} params")

    # Compare with full model
    print("\n" + "=" * 50)
    print("🔍 FULL MODEL VERIFICATION")
    print("=" * 50)

    x = torch.randn(input_shape).to(device)
    full_model_result = count_module_flops(model, x, "Full CMFDNet")

    if full_model_result:
        print(f"Full model FLOPs: {full_model_result['flops_str']}")
        print(f"Full model params: {full_model_result['params_str']}")

        if total_flops > 0:
            ratio = full_model_result["flops"] / total_flops
            print(".2f")
            if abs(ratio - 1.0) > 0.1:  # More than 10% difference
                print("⚠️  Significant difference - check module analysis")

    print("\n" + "=" * 100)
    print("✅ FLOPs ANALYSIS COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        "CMFD Model FLOPs Calculator", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    # Try to load config, use default if not found
    try:
        with open(args.cfg_path, "r+", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        Path(config.get("training", {}).get("model_dir", "models")).mkdir(
            parents=True, exist_ok=True
        )
    except FileNotFoundError:
        print(f"Config file {args.cfg_path} not found, using default configuration")
        config = {
            "data": {"image_size": args.image_size},
            "model": {
                "encoder_out_ch": 256,
                "gce_embed_dim": 256,
                "gce_patch_size": 9,
                "aspp_out_ch": 128,
                "fuse_out_ch": 256,
                "T_affine": 4,
                "num_cls": 2,
            },
        }

    main(args, config)
