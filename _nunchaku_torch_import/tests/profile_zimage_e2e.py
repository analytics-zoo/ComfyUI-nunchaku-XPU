"""Z-Image e2e for profiling with unitrace.

Runs a single denoise step (not full pipeline) to focus on transformer kernels.
Text encoder is skipped — we profile only the transformer forward pass.
"""
import sys, time, torch
sys.path.insert(0, "src")
from nunchaku_torch import NunchakuZImageTransformer2DModel
from nunchaku_torch.models.linear import SVDQW4A4Linear
from nunchaku_torch.models.transformers.transformer_zimage import NunchakuZImageFusedModule

quant_path = "/LLM/models/nunchaku-z-image-turbo/svdq-int4_r32-z-image-turbo.safetensors"
device = torch.device("xpu:0")

print("Loading Z-Image transformer...")
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    quant_path, device="cpu", torch_dtype=torch.bfloat16,
)

# Move quantized weights to XPU
for m in transformer.modules():
    if isinstance(m, SVDQW4A4Linear):
        for pn, p in list(m.named_parameters()):
            setattr(m, pn, torch.nn.Parameter(p.to(device), requires_grad=False))
    elif isinstance(m, NunchakuZImageFusedModule):
        for attr in dir(m):
            if attr.startswith("qkv_") or attr.startswith("norm_"):
                val = getattr(m, attr)
                if isinstance(val, torch.nn.Parameter):
                    setattr(m, attr, torch.nn.Parameter(val.to(device), requires_grad=False))
torch.xpu.synchronize()
print(f"XPU mem: {torch.xpu.memory_allocated()/1e9:.2f} GB")

# Prepare input (256x256 image equivalent)
C, F, H, W = 16, 1, 32, 32  # 256x256
x = [torch.randn(C, F, H, W, dtype=torch.bfloat16)]
t_step = torch.tensor([500.0], dtype=torch.bfloat16)
cap_feats = [torch.randn(64, 2560, dtype=torch.bfloat16)]

# Warmup
print("Warmup...")
with torch.no_grad():
    _ = transformer(x, t_step, cap_feats)
torch.xpu.synchronize()

# Profiled run
print("Profiled forward pass...")
torch.xpu.synchronize()
t0 = time.time()
with torch.no_grad():
    out = transformer(x, t_step, cap_feats)
torch.xpu.synchronize()
elapsed = time.time() - t0

s = out.sample[0] if isinstance(out.sample, list) else out.sample
print(f"Output: {s.shape}, time={elapsed:.3f}s")
print("Done")
