"""Profile Z-Image AFTER warmup — only the steady-state forward pass."""
import sys, time, torch
sys.path.insert(0, "src")
from nunchaku_torch import NunchakuZImageTransformer2DModel

quant_path = "/LLM/models/nunchaku-z-image-turbo/svdq-int4_r32-z-image-turbo.safetensors"
device = torch.device("xpu:0")

print("Loading...")
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    quant_path, device="cpu", torch_dtype=torch.bfloat16,
)
transformer = transformer.to(device)

C, F, H, W = 16, 1, 32, 32
x = [torch.randn(C, F, H, W, dtype=torch.bfloat16, device=device)]
t_step = torch.tensor([500.0], dtype=torch.bfloat16, device=device)
cap_feats = [torch.randn(64, 2560, dtype=torch.bfloat16, device=device)]

# Warmup (2 runs to fill all caches)
print("Warmup...")
with torch.no_grad():
    _ = transformer(x, t_step, cap_feats)
    _ = transformer(x, t_step, cap_feats)
torch.xpu.synchronize()

# Profiled run
print("Profiled run...")
torch.xpu.synchronize()
t0 = time.time()
with torch.no_grad():
    out = transformer(x, t_step, cap_feats)
torch.xpu.synchronize()
print(f"Time: {(time.time()-t0)*1000:.1f}ms")
