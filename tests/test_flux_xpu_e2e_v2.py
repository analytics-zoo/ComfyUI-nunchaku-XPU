"""FLUX XPU e2e v2: use exactly the same pipeline as CPU W4A16 baseline,
but with _forward_xpu enabled via XPU weights."""
import sys, time, types, os
sys.path.insert(0, "src")
import torch, torch.nn.functional as F
from nunchaku_torch.models.transformers.transformer_flux import NunchakuFluxTransformer2DModel
from nunchaku_torch.models.linear import SVDQW4A4Linear

quant_path = "/LLM/models/nunchaku-flux.1-schnell/svdq-int4_r32-flux.1-schnell.safetensors"
base_model = "/LLM/models/FLUX.1-schnell"
device = torch.device("xpu:0")
dtype = torch.bfloat16

print("=== FLUX XPU E2E v2 ===")

print("[1/3] Loading transformer + moving SVDQ to XPU...")
transformer = NunchakuFluxTransformer2DModel.from_pretrained(
    quant_path, device="cpu", torch_dtype=dtype, cpu_kernel_layout=True,
)
for m in transformer.modules():
    if isinstance(m, SVDQW4A4Linear):
        for pn, p in list(m.named_parameters()):
            setattr(m, pn, torch.nn.Parameter(p.to(device), requires_grad=False))
torch.xpu.synchronize()
print(f"  XPU mem: {torch.xpu.memory_allocated()/1e9:.2f} GB")

print("[2/3] Loading pipeline...")
from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True)
tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", local_files_only=True)
text_encoder_2 = T5EncoderModel.from_pretrained(base_model, subfolder="text_encoder_2", torch_dtype=dtype, local_files_only=True)
tokenizer_2 = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_2", local_files_only=True)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler", local_files_only=True)

pipe = FluxPipeline(
    transformer=transformer, vae=vae,
    text_encoder=text_encoder, text_encoder_2=text_encoder_2,
    tokenizer=tokenizer, tokenizer_2=tokenizer_2,
    scheduler=scheduler,
)

print("[3/3] Generating (256x256, 4 steps)...")
t0 = time.time()
with torch.no_grad():
    result = pipe(
        prompt="a cute cat sitting on a windowsill",
        height=256, width=256,
        num_inference_steps=4,
        guidance_scale=3.5,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )
image = result.images[0]
output_path = "/LLM/flux_xpu_e2e_v2.png"
image.save(output_path)

import numpy as np
img_arr = np.array(image)
print(f"\nGenerated in {time.time()-t0:.1f}s")
print(f"Image: {image.size}, pixel range [{img_arr.min()}, {img_arr.max()}], mean {img_arr.mean():.1f}")
print(f"Saved to {output_path}")
