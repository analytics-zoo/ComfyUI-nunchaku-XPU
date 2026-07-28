"""Kernel-level parity test: verify each XPU kernel matches CPU exactly.

Tests with REAL model weights (not random) to catch format/layout issues.
"""
import sys, torch
sys.path.insert(0, "src")
from nunchaku_torch import NunchakuZImageTransformer2DModel
from nunchaku_torch.models.linear import SVDQW4A4Linear
from nunchaku_torch.ops.cpu_ops import dequantize_w4a4
from omni_xpu_kernel import svdq as omni_svdq

quant_path = "/LLM/models/nunchaku-z-image-turbo/svdq-int4_r32-z-image-turbo.safetensors"
device = torch.device("xpu:0")

print("Loading model...")
t = NunchakuZImageTransformer2DModel.from_pretrained(
    quant_path, device="cpu", torch_dtype=torch.bfloat16,
)

# Get a real SVDQ layer
layer = t.noise_refiner[0].attention.to_out[0]
print(f"Layer: SVDQW4A4Linear({layer.in_features}->{layer.out_features})")
print(f"  qweight: {layer.qweight.shape} {layer.qweight.dtype}")
print(f"  smooth: [{layer.smooth_factor.min():.3f}, {layer.smooth_factor.max():.3f}]")

torch.manual_seed(42)
x = torch.randn(32, layer.in_features, dtype=torch.bfloat16)

print("\n=== Test 1: Weight dequantization ===")
w_cpu = dequantize_w4a4(layer.qweight, layer.wscales, 64, True)
w_xpu = omni_svdq.dequantize_w4(
    layer.qweight.to(device).view(torch.uint8),
    layer.wscales.to(device),
    torch.bfloat16,
).cpu()
# Convert CPU result to bf16 for fair comparison
w_cpu_bf16 = w_cpu.to(torch.bfloat16)
diff_w = (w_cpu_bf16.float() - w_xpu.float()).abs()
print(f"  CPU bf16 vs XPU bf16: max={diff_w.max():.6f}, exact_match={torch.equal(w_cpu_bf16, w_xpu)}")

print("\n=== Test 2: Smooth factor ===")
# CPU: x.float() / smooth.float() -> bf16
xs_cpu = (x.float() / layer.smooth_factor.float()).to(torch.bfloat16)
# XPU: fused_smooth_mul_convert(bf16, f16_rcp) -> f16 -> bf16
rcp = (1.0 / layer.smooth_factor.float()).to(torch.float16).to(device)
xs_xpu_f16 = omni_svdq.fused_smooth_mul_convert(x.to(device), rcp)
xs_xpu_bf16 = xs_xpu_f16.to(torch.bfloat16).cpu()
diff_s = (xs_cpu.float() - xs_xpu_bf16.float()).abs()
rel_s = diff_s.mean() / xs_cpu.float().abs().mean()
print(f"  rel_diff={rel_s:.6f} ({rel_s*100:.4f}%)")
print(f"  This is the bf16→f16→bf16 truncation error")

# Direct comparison: CPU bf16 vs CPU (div, bf16)
xs_cpu_direct = (x.float() / layer.smooth_factor.float())  # float32
xs_via_f16 = xs_xpu_f16.cpu().float()  # was f16
diff_s2 = (xs_cpu_direct - xs_via_f16).abs()
rel_s2 = diff_s2.mean() / xs_cpu_direct.abs().mean()
print(f"  float32 vs f16 smooth: rel={rel_s2:.6f} ({rel_s2*100:.4f}%)")

print("\n=== Test 3: GEMM (smooth → GEMM, no LoRA/bias) ===")
# CPU: float32 smooth → float32 matmul with dequant weights
y_cpu = (xs_cpu.float() @ w_cpu.float().T).to(torch.bfloat16)
# XPU: bf16 smooth → oneDNN INT4 GEMM
u4, sf16 = omni_svdq.prepare_onednn_weights(
    layer.qweight.to(device).view(torch.uint8),
    layer.wscales.to(device),
)
y_xpu = omni_svdq.onednn_int4_gemm_preconverted(
    xs_xpu_bf16.to(device), u4, sf16
).cpu().to(torch.bfloat16)
diff_g = (y_cpu.float() - y_xpu.float()).abs()
rel_g = diff_g.mean() / y_cpu.float().abs().mean()
print(f"  CPU dequant+matmul vs XPU oneDNN: rel={rel_g:.6f} ({rel_g*100:.4f}%)")

# Same input comparison: use SAME smoothed input for both
y_xpu_same = omni_svdq.onednn_int4_gemm_preconverted(
    xs_cpu.to(device), u4, sf16
).cpu().to(torch.bfloat16)
diff_g2 = (y_cpu.float() - y_xpu_same.float()).abs()
rel_g2 = diff_g2.mean() / y_cpu.float().abs().mean()
print(f"  Same input: CPU matmul vs XPU oneDNN: rel={rel_g2:.6f} ({rel_g2*100:.4f}%)")

print("\n=== Test 4: Full _forward_xpu vs CPU W4A16 ===")
# CPU W4A16 (reference)
x_3d = x.unsqueeze(0)  # [1, 32, 3840]
x_flat = x
xs_full = x_flat.float() / layer.smooth_factor.float()
y_full_cpu = (xs_full @ w_cpu.float().T)
y_full_cpu += (x_flat.float() @ layer.proj_down.float()) @ layer.proj_up.float().T
if layer.bias is not None:
    y_full_cpu += layer.bias.float()
y_full_cpu = y_full_cpu.to(torch.bfloat16)

# XPU _forward_xpu
layer_xpu = layer.to(device)
with torch.no_grad():
    y_full_xpu = layer_xpu(x_3d).squeeze(0).cpu()
diff_f = (y_full_cpu.float() - y_full_xpu.float()).abs()
rel_f = diff_f.mean() / y_full_cpu.float().abs().mean()
print(f"  _forward_xpu vs CPU W4A16: rel={rel_f:.6f} ({rel_f*100:.4f}%)")

print("\n=== Summary ===")
print(f"  Weight dequant: {'EXACT' if torch.equal(w_cpu_bf16, w_xpu) else 'DIFF'}")
print(f"  Smooth (bf16→f16→bf16): {rel_s*100:.4f}%")
print(f"  GEMM (same input): {rel_g2*100:.4f}%")
print(f"  GEMM (different smooth): {rel_g*100:.4f}%")
print(f"  Full forward: {rel_f*100:.4f}%")
