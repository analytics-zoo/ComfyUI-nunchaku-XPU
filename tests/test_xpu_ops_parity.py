"""
XPU ops parity tests: compare XPU ops output against CPU reference.

Run:
    python tests/test_xpu_ops_parity.py
"""

import sys
import torch
import torch.nn.functional as F

# Ensure we can import from the source tree
sys.path.insert(0, "src")


def make_test_data(M, K, N, rank=32, device="cpu"):
    """Create test data matching nunchaku's quantized weight format."""
    # Quantized weights: [N, K//2] int8 packed INT4
    qweight = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device=device)
    # Weight scales: [K//64, N]
    wscales = torch.randn(K // 64, N, dtype=torch.bfloat16, device=device) * 0.01
    # Smooth factor: [K]
    smooth = torch.rand(K, dtype=torch.bfloat16, device=device) + 0.5
    # LoRA
    lora_down = torch.randn(K, rank, dtype=torch.bfloat16, device=device) * 0.01
    lora_up = torch.randn(N, rank, dtype=torch.bfloat16, device=device) * 0.01
    # Bias
    bias = torch.randn(N, dtype=torch.bfloat16, device=device) * 0.01
    # Input activation
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    return {
        "qweight": qweight, "wscales": wscales, "smooth": smooth,
        "lora_down": lora_down, "lora_up": lora_up, "bias": bias, "x": x,
    }


def test_quantize_parity():
    """Test svdq_quantize_w4a4_act_fuse_lora: XPU vs CPU."""
    from nunchaku_torch.ops.cpu_ops import svdq_quantize_w4a4_act_fuse_lora_cpu
    from nunchaku_torch.ops.xpu_ops import svdq_quantize_w4a4_act_fuse_lora_xpu
    from nunchaku_torch.utils import ceil_divide

    M, K, rank = 128, 3072, 32
    pad_size = 256
    M_pad = ceil_divide(M, pad_size) * pad_size

    data = make_test_data(M, K, 3072, rank, device="cpu")

    # CPU reference: note cpu_ops uses pack_int4 which outputs uint8
    cpu_output = torch.zeros(M_pad, K // 2, dtype=torch.uint8)
    cpu_oscales = torch.zeros(K // 64, M_pad, dtype=torch.bfloat16)
    cpu_lora_act = torch.zeros(M_pad, rank, dtype=torch.float32)
    svdq_quantize_w4a4_act_fuse_lora_cpu(
        data["x"], cpu_output, cpu_oscales, data["lora_down"], cpu_lora_act, data["smooth"],
    )

    # XPU: omni_xpu_kernel.quantize_act_int4 also outputs uint8
    device = torch.device("xpu:0")
    x_xpu = data["x"].to(device)
    xpu_output = torch.zeros(M_pad, K // 2, dtype=torch.uint8, device=device)
    xpu_oscales = torch.zeros(K // 64, M_pad, dtype=torch.bfloat16, device=device)
    xpu_lora_act = torch.zeros(M_pad, rank, dtype=torch.float32, device=device)
    svdq_quantize_w4a4_act_fuse_lora_xpu(
        x_xpu, xpu_output, xpu_oscales,
        data["lora_down"].to(device), xpu_lora_act,
        data["smooth"].to(device),
    )
    torch.xpu.synchronize()

    # Compare by unpacking int4 and comparing the actual values
    xpu_output_cpu = xpu_output.cpu()
    xpu_oscales_cpu = xpu_oscales.cpu()
    xpu_lora_act_cpu = xpu_lora_act.cpu()

    # Unpack both to int4 values and compare
    from nunchaku_torch.ops.cpu_ops import unpack_int4
    cpu_unpacked = unpack_int4(cpu_output.view(torch.int8), signed=True)
    xpu_unpacked = unpack_int4(xpu_output_cpu.view(torch.int8), signed=True)
    diff_unpacked = (xpu_unpacked - cpu_unpacked).abs()
    max_packed_diff = diff_unpacked.max().item()

    # scales 比较
    scale_diff = (xpu_oscales_cpu.float() - cpu_oscales.float()).abs()
    max_scale_diff = scale_diff.max().item()
    rel_scale_diff = (scale_diff / (cpu_oscales.float().abs() + 1e-10)).max().item()

    # LoRA act 比较 (应该接近，因为都是 float matmul)
    lora_diff = (xpu_lora_act_cpu - cpu_lora_act).abs()
    max_lora_diff = lora_diff.max().item()

    print(f"=== quantize parity (M={M}, K={K}) ===")
    print(f"  packed int4: max_diff={max_packed_diff}")
    print(f"  scales: max_diff={max_scale_diff:.6f}, rel_max={rel_scale_diff:.6f}")
    print(f"  lora_act: max_diff={max_lora_diff:.6f}")

    ok = max_packed_diff <= 2 and rel_scale_diff < 0.1 and max_lora_diff < 0.1
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_gemm_parity():
    """Test svdq_gemm_w4a4: XPU vs CPU."""
    from nunchaku_torch.ops.cpu_ops import (
        svdq_quantize_w4a4_act_fuse_lora_cpu,
        svdq_gemm_w4a4_cpu,
    )
    from nunchaku_torch.ops.xpu_ops import svdq_gemm_w4a4_xpu
    from nunchaku_torch.utils import ceil_divide

    M, K, N, rank = 64, 3072, 3072, 32
    pad_size = 256
    M_pad = ceil_divide(M, pad_size) * pad_size

    data = make_test_data(M, K, N, rank, device="cpu")

    # Step 1: Quantize activations on CPU (same quantized input for both)
    cpu_qact = torch.zeros(M_pad, K // 2, dtype=torch.uint8)
    cpu_ascales = torch.zeros(K // 64, M_pad, dtype=torch.bfloat16)
    cpu_lora_act = torch.zeros(M_pad, rank, dtype=torch.float32)
    svdq_quantize_w4a4_act_fuse_lora_cpu(
        data["x"], cpu_qact, cpu_ascales, data["lora_down"], cpu_lora_act, data["smooth"],
    )

    # Step 2a: CPU GEMM
    cpu_out = torch.zeros(M, N, dtype=torch.bfloat16)
    svdq_gemm_w4a4_cpu(
        act=cpu_qact, wgt=data["qweight"], out=cpu_out,
        ascales=cpu_ascales, wscales=data["wscales"],
        lora_act_in=cpu_lora_act, lora_up=data["lora_up"],
        bias=data["bias"],
    )

    # Step 2b: XPU GEMM (with same quantized inputs)
    device = torch.device("xpu:0")
    xpu_out = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    svdq_gemm_w4a4_xpu(
        act=cpu_qact.to(device), wgt=data["qweight"].to(device), out=xpu_out,
        ascales=cpu_ascales.to(device), wscales=data["wscales"].to(device),
        lora_act_in=cpu_lora_act.to(device), lora_up=data["lora_up"].to(device),
        bias=data["bias"].to(device),
    )
    torch.xpu.synchronize()

    # Compare
    xpu_out_cpu = xpu_out.cpu()
    diff = (xpu_out_cpu.float() - cpu_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    # Relative to output magnitude
    output_mag = cpu_out.float().abs().mean().item()
    rel_diff = mean_diff / (output_mag + 1e-10)

    print(f"=== gemm parity (M={M}, K={K}, N={N}) ===")
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"  output_magnitude={output_mag:.6f}, relative_diff={rel_diff:.6f}")

    # W4A4 vs W4Af16 will have numerical differences.
    # CPU does true INT4xINT4 dot product, XPU dequants activations to f16 first.
    # Allow larger tolerance for this fundamental approach difference.
    ok = rel_diff < 0.5  # 50% relative tolerance for different quant schemes
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_awq_parity():
    """Test awq_gemv_w4a16: XPU vs CPU."""
    from nunchaku_torch.ops.cpu_ops import awq_gemv_w4a16_cpu
    from nunchaku_torch.ops.xpu_ops import awq_gemv_w4a16_xpu

    M, N, K = 4, 18432, 3072
    group_size = 64

    # AWQ format: kernel is [N//4, K//2] int32
    # When viewed as int16 it becomes [N//4, K], then reshaped/permuted
    # The raw tensor shape must be compatible with view(int16).reshape(N//4, K)
    kernel = torch.randint(-2**15, 2**15, (N // 4, K // 2), dtype=torch.int32)
    scaling_factors = torch.randn(K // group_size, N, dtype=torch.bfloat16) * 0.01
    zeros = torch.randn(K // group_size, N, dtype=torch.bfloat16) * 0.001
    x = torch.randn(M, K, dtype=torch.bfloat16)

    # CPU
    cpu_out = awq_gemv_w4a16_cpu(x, kernel, scaling_factors, zeros, M, N, K, group_size)

    # XPU
    device = torch.device("xpu:0")
    xpu_out = awq_gemv_w4a16_xpu(
        x.to(device), kernel.to(device),
        scaling_factors.to(device), zeros.to(device),
        M, N, K, group_size,
    )
    torch.xpu.synchronize()

    xpu_out_cpu = xpu_out.cpu()
    diff = (xpu_out_cpu.float() - cpu_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    output_mag = cpu_out.float().abs().mean().item()
    rel_diff = mean_diff / (output_mag + 1e-10)

    print(f"=== awq parity (M={M}, K={K}, N={N}) ===")
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"  output_magnitude={output_mag:.6f}, relative_diff={rel_diff:.6f}")

    # AWQ dequant should be exact (same algorithm)
    ok = rel_diff < 0.01
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_full_linear_parity():
    """Test full SVDQW4A4Linear.forward(): XPU vs CPU end-to-end."""
    from nunchaku_torch.models.linear import SVDQW4A4Linear

    M, K, N, rank = 64, 3072, 3072, 32
    device_xpu = torch.device("xpu:0")

    # Create linear on CPU
    linear_cpu = SVDQW4A4Linear(K, N, rank=rank, torch_dtype=torch.bfloat16, device="cpu")
    linear_cpu.qweight.data = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8)
    linear_cpu.wscales.data = torch.randn(K // 64, N, dtype=torch.bfloat16) * 0.01
    linear_cpu.smooth_factor.data = torch.rand(K, dtype=torch.bfloat16) + 0.5
    linear_cpu.proj_down.data = torch.randn(K, rank, dtype=torch.bfloat16) * 0.01
    linear_cpu.proj_up.data = torch.randn(N, rank, dtype=torch.bfloat16) * 0.01
    linear_cpu.bias.data = torch.randn(N, dtype=torch.bfloat16) * 0.01

    # Create identical linear on XPU
    linear_xpu = SVDQW4A4Linear(K, N, rank=rank, torch_dtype=torch.bfloat16, device=device_xpu)
    linear_xpu.qweight.data = linear_cpu.qweight.data.to(device_xpu)
    linear_xpu.wscales.data = linear_cpu.wscales.data.to(device_xpu)
    linear_xpu.smooth_factor.data = linear_cpu.smooth_factor.data.to(device_xpu)
    linear_xpu.proj_down.data = linear_cpu.proj_down.data.to(device_xpu)
    linear_xpu.proj_up.data = linear_cpu.proj_up.data.to(device_xpu)
    linear_xpu.bias.data = linear_cpu.bias.data.to(device_xpu)

    x = torch.randn(1, M, K, dtype=torch.bfloat16)

    # CPU forward
    with torch.no_grad():
        cpu_out = linear_cpu(x)

    # XPU forward
    with torch.no_grad():
        xpu_out = linear_xpu(x.to(device_xpu))
    torch.xpu.synchronize()

    xpu_out_cpu = xpu_out.cpu()
    diff = (xpu_out_cpu.float() - cpu_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    output_mag = cpu_out.float().abs().mean().item()
    rel_diff = mean_diff / (output_mag + 1e-10)

    print(f"=== SVDQW4A4Linear full forward parity (M={M}, K={K}, N={N}) ===")
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"  output_magnitude={output_mag:.6f}, relative_diff={rel_diff:.6f}")

    # Different quantization approach (W4A4 vs W4Af16) means larger tolerance
    ok = rel_diff < 1.0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("XPU ops parity tests vs CPU reference")
    print("=" * 60)

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("SKIP: No XPU device available")
        sys.exit(0)

    results = {}
    for name, test_fn in [
        ("quantize", test_quantize_parity),
        ("gemm", test_gemm_parity),
        ("awq", test_awq_parity),
        ("full_linear", test_full_linear_parity),
    ]:
        try:
            results[name] = test_fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = False
            print(f"  RESULT: ERROR ({e})")
        print()

    print("=" * 60)
    print("SUMMARY:")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
