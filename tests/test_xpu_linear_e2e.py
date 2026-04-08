"""
End-to-end test: SVDQW4A4Linear and AWQW4A16Linear on XPU via nunchaku_torch.

Tests the full ops dispatch chain: nunchaku_torch -> ops dispatcher -> xpu_ops -> omni_xpu_kernel
"""
import sys
import time
sys.path.insert(0, "src")

import torch


def test_svdq_linear_dispatch():
    """Test that SVDQW4A4Linear on XPU auto-dispatches to xpu_ops."""
    from nunchaku_torch.models.linear import SVDQW4A4Linear

    device = torch.device("xpu:0")
    K, N, rank = 3072, 3072, 32

    linear = SVDQW4A4Linear(K, N, rank=rank, torch_dtype=torch.bfloat16, device=device)
    linear.qweight.data = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device=device)
    linear.wscales.data = torch.randn(K // 64, N, dtype=torch.bfloat16, device=device) * 0.01
    linear.smooth_factor.data = torch.rand(K, dtype=torch.bfloat16, device=device) + 0.5
    linear.proj_down.data = torch.randn(K, rank, dtype=torch.bfloat16, device=device) * 0.01
    linear.proj_up.data = torch.randn(N, rank, dtype=torch.bfloat16, device=device) * 0.01
    linear.bias.data = torch.randn(N, dtype=torch.bfloat16, device=device) * 0.01

    x = torch.randn(1, 64, K, dtype=torch.bfloat16, device=device)

    # Warmup
    with torch.no_grad():
        _ = linear(x)
    torch.xpu.synchronize()

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            y = linear(x)
    torch.xpu.synchronize()
    elapsed = (time.time() - start) / 10

    print(f"SVDQW4A4Linear [{K}->{N}]: shape={y.shape}, time={elapsed*1000:.2f}ms")
    assert y.shape == (1, 64, N), f"Bad shape: {y.shape}"
    assert y.device.type == "xpu", f"Bad device: {y.device}"
    print("  PASS")
    return True


def test_awq_linear_dispatch():
    """Test that AWQW4A16Linear on XPU auto-dispatches to xpu_ops."""
    from nunchaku_torch.models.linear import AWQW4A16Linear

    device = torch.device("xpu:0")
    K, N = 3072, 18432
    group_size = 64

    linear = AWQW4A16Linear(K, N, group_size=group_size, torch_dtype=torch.bfloat16, device=device)
    # AWQ qweight: [N//4, K//2] int32
    linear.qweight.data = torch.randint(-2**15, 2**15, (N // 4, K // 2), dtype=torch.int32, device=device)
    linear.wscales.data = torch.randn(K // group_size, N, dtype=torch.bfloat16, device=device) * 0.01
    linear.wzeros.data = torch.randn(K // group_size, N, dtype=torch.bfloat16, device=device) * 0.001
    linear.bias.data = torch.randn(N, dtype=torch.bfloat16, device=device) * 0.01

    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        y = linear(x)
    torch.xpu.synchronize()

    print(f"AWQW4A16Linear [{K}->{N}]: shape={y.shape}")
    assert y.shape == (1, N), f"Bad shape: {y.shape}"
    assert y.device.type == "xpu", f"Bad device: {y.device}"
    print("  PASS")
    return True


def test_fused_gelu_mlp_dispatch():
    """Test fused_gelu_mlp on XPU."""
    from nunchaku_torch.models.linear import SVDQW4A4Linear
    from nunchaku_torch.ops.fused import fused_gelu_mlp

    device = torch.device("xpu:0")
    K, hidden, N, rank = 3072, 12288, 3072, 32

    fc1 = SVDQW4A4Linear(K, hidden, rank=rank, torch_dtype=torch.bfloat16, device=device)
    fc1.qweight.data = torch.randint(-128, 127, (hidden, K // 2), dtype=torch.int8, device=device)
    fc1.wscales.data = torch.randn(K // 64, hidden, dtype=torch.bfloat16, device=device) * 0.01
    fc1.smooth_factor.data = torch.rand(K, dtype=torch.bfloat16, device=device) + 0.5
    fc1.proj_down.data = torch.randn(K, rank, dtype=torch.bfloat16, device=device) * 0.01
    fc1.proj_up.data = torch.randn(hidden, rank, dtype=torch.bfloat16, device=device) * 0.01
    fc1.bias.data = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.01

    fc2 = SVDQW4A4Linear(hidden, N, rank=rank, torch_dtype=torch.bfloat16, device=device)
    fc2.qweight.data = torch.randint(-128, 127, (N, hidden // 2), dtype=torch.int8, device=device)
    fc2.wscales.data = torch.randn(hidden // 64, N, dtype=torch.bfloat16, device=device) * 0.01
    fc2.smooth_factor.data = torch.rand(hidden, dtype=torch.bfloat16, device=device) + 0.5
    fc2.proj_down.data = torch.randn(hidden, rank, dtype=torch.bfloat16, device=device) * 0.01
    fc2.proj_up.data = torch.randn(N, rank, dtype=torch.bfloat16, device=device) * 0.01
    fc2.bias.data = torch.randn(N, dtype=torch.bfloat16, device=device) * 0.01

    x = torch.randn(1, 64, K, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        y = fused_gelu_mlp(x, fc1, fc2)
    torch.xpu.synchronize()

    print(f"fused_gelu_mlp [{K}->{hidden}->{N}]: shape={y.shape}")
    assert y.shape == (1, 64, N), f"Bad shape: {y.shape}"
    assert y.device.type == "xpu", f"Bad device: {y.device}"
    print("  PASS")
    return True


if __name__ == "__main__":
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("SKIP: No XPU device")
        sys.exit(0)

    print("=" * 60)
    print("nunchaku_torch XPU dispatch e2e tests")
    print("=" * 60)

    results = {}
    for name, fn in [
        ("svdq_linear", test_svdq_linear_dispatch),
        ("awq_linear", test_awq_linear_dispatch),
        ("fused_gelu_mlp", test_fused_gelu_mlp_dispatch),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = False
        print()

    print("SUMMARY:")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
