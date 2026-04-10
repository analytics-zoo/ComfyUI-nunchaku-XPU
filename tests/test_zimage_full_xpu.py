"""Z-Image: entire transformer on XPU (no per-layer CPU↔XPU transfer).

This should eliminate the 67% memory copy overhead seen in profiling.
"""
import sys, time, torch
sys.path.insert(0, "src")
from nunchaku_torch import NunchakuZImageTransformer2DModel

quant_path = "/LLM/models/nunchaku-z-image-turbo/svdq-int4_r32-z-image-turbo.safetensors"
device = torch.device("xpu:0")

print("=== Z-Image Full XPU ===")

print("[1/3] Loading on CPU...")
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    quant_path, device="cpu", torch_dtype=torch.bfloat16,
)

print("[2/3] Moving entire model to XPU...")
t0 = time.time()
transformer = transformer.to(device)
torch.xpu.synchronize()
print(f"  Moved in {time.time()-t0:.1f}s, XPU mem: {torch.xpu.memory_allocated()/1e9:.2f} GB")

# Input on XPU directly
C, F, H, W = 16, 1, 32, 32
x = [torch.randn(C, F, H, W, dtype=torch.bfloat16, device=device)]
t_step = torch.tensor([500.0], dtype=torch.bfloat16, device=device)
cap_feats = [torch.randn(64, 2560, dtype=torch.bfloat16, device=device)]

# Warmup
print("[3/3] Forward pass...")
with torch.no_grad():
    _ = transformer(x, t_step, cap_feats)
torch.xpu.synchronize()

# Timed run
torch.xpu.synchronize()
t0 = time.time()
with torch.no_grad():
    out = transformer(x, t_step, cap_feats)
torch.xpu.synchronize()
elapsed = time.time() - t0

s = out.sample[0] if isinstance(out.sample, list) else out.sample
print(f"  Output: {s.shape}, nan={torch.isnan(s).any()}, range=[{s.min():.2f},{s.max():.2f}]")
print(f"  Time: {elapsed*1000:.1f}ms")
print(f"  XPU mem: {torch.xpu.memory_allocated()/1e9:.2f} GB")
