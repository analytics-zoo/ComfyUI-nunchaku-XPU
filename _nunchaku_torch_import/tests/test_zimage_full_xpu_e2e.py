"""Z-Image full XPU e2e: transformer entirely on XPU, pipeline manages device transfer."""
import sys, time
sys.path.insert(0, "src")
import torch
from nunchaku_torch import NunchakuZImageTransformer2DModel

quant_path = "/LLM/models/nunchaku-z-image-turbo/svdq-int4_r32-z-image-turbo.safetensors"
base_model = "/LLM/models/Z-Image-Turbo"
device = torch.device("xpu:0")
dtype = torch.bfloat16

print("=== Z-Image Full XPU E2E ===")

print("[1/4] Loading transformer on XPU...")
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    quant_path, device="cpu", torch_dtype=dtype,
)
transformer = transformer.to(device)
torch.xpu.synchronize()
print(f"  XPU mem: {torch.xpu.memory_allocated()/1e9:.2f} GB")

print("[2/4] Loading pipeline on CPU...")
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained(
    base_model, transformer=transformer, torch_dtype=dtype, local_files_only=True,
)
# text_encoder, vae, tokenizer stay on CPU
# transformer is already on XPU

# Monkey-patch transformer forward to handle CPU↔XPU at pipeline boundary
_orig_forward = transformer.forward
def _boundary_forward(*args, **kwargs):
    """Move inputs to XPU, run forward, move outputs back to CPU."""
    def _to_xpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, list):
            return [_to_xpu(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(_to_xpu(x) for x in obj)
        return obj

    new_args = [_to_xpu(a) for a in args]
    new_kwargs = {k: _to_xpu(v) for k, v in kwargs.items()}

    result = _orig_forward(*new_args, **new_kwargs)

    # Move ALL result tensors back to CPU
    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, list):
            return [_to_cpu(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(_to_cpu(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        return obj

    if hasattr(result, 'sample'):
        result['sample'] = _to_cpu(result['sample'])
    elif isinstance(result, (tuple, list)):
        result = _to_cpu(result)
    return result

pipe.transformer.forward = _boundary_forward

print("[3/4] Warmup...")
# Quick warmup
with torch.no_grad():
    warmup_x = [torch.randn(16, 1, 8, 8, dtype=dtype, device=device)]
    warmup_t = torch.tensor([500.0], dtype=dtype, device=device)
    warmup_c = [torch.randn(16, 2560, dtype=dtype, device=device)]
    _ = _orig_forward(warmup_x, warmup_t, warmup_c)
    torch.xpu.synchronize()
print("  Warmup done")

print("[4/4] Generating (256x256, 4 steps)...")
t0 = time.time()
with torch.no_grad():
    result = pipe(
        prompt="a cute cat sitting on a windowsill, highly detailed",
        height=256, width=256,
        num_inference_steps=4,
        guidance_scale=3.5,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )
elapsed = time.time() - t0

image = result.images[0]
output_path = "/LLM/zimage_full_xpu_e2e.png"
image.save(output_path)

import numpy as np
img_arr = np.array(image)
print(f"\nGenerated in {elapsed:.1f}s")
print(f"Image: {image.size}, pixel [{img_arr.min()},{img_arr.max()}], mean {img_arr.mean():.1f}")
print(f"Saved to {output_path}")
