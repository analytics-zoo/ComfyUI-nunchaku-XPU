from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

from .device import default_dtype, resolve_device


@dataclass(slots=True)
class GenerationConfig:
    quant_path: str
    base_model: str
    prompt: str
    device: str = "auto"
    height: int = 256
    width: int = 256
    steps: int = 9
    guidance: float = 0.0
    seed: int = 12345


def load_pipeline(
    config: GenerationConfig,
) -> tuple[ZImagePipeline, torch.device, torch.dtype]:
    from .models.transformers import transformer_zimage
    from .models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel

    device = resolve_device(config.device)
    dtype = default_dtype(device)
    transformer_zimage.get_precision = lambda *args, **kwargs: "int4"

    transformer_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if device.type != "cpu":
        transformer_kwargs["device"] = str(device)

    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        config.quant_path, **transformer_kwargs
    )
    pipe = ZImagePipeline.from_pretrained(
        config.base_model,
        transformer=transformer,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        local_files_only=True,
    )
    pipe = pipe.to(str(device))
    return pipe, device, dtype


def _build_generation_kwargs(
    pipe: ZImagePipeline,
    config: GenerationConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    do_classifier_free_guidance = config.guidance > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=config.prompt,
        device=device,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=None,
        max_sequence_length=512,
    )
    num_channels_latents = pipe.transformer.config.in_channels
    generator = torch.Generator(device=str(device)).manual_seed(config.seed)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=config.height,
        width=config.width,
        dtype=dtype,
        device=str(device),
        generator=generator,
        latents=None,
    )
    return {
        "prompt": None,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "height": config.height,
        "width": config.width,
        "num_inference_steps": config.steps,
        "guidance_scale": config.guidance,
        "generator": None,
        "latents": latents,
    }


def generate_image(config: GenerationConfig):
    pipe, device, dtype = load_pipeline(config)
    generation_kwargs = _build_generation_kwargs(pipe, config, device, dtype)
    result = pipe(**generation_kwargs)
    images: Any = getattr(result, "images", None)
    if images is None:
        raise RuntimeError("ZImage pipeline result did not contain images.")
    return images[0]


def save_image(config: GenerationConfig, output: str | Path):
    image = generate_image(config)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
