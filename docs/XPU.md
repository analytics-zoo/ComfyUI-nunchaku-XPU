# Intel XPU Support

This repository adds Intel XPU support through its bundled `nunchaku_torch`
runtime, [Comfy Kitchen XPU](https://github.com/xiangyuT/comfy-kitchen-xpu),
and the
`omni_xpu_kernel` native companion package from
[Intel llm-scaler](https://github.com/intel/llm-scaler).

The integration boundary is:

```text
ComfyUI node -> bundled nunchaku_torch model lifecycle
             -> Comfy Kitchen preparation/dispatch/fallback
             -> omni_xpu_kernel ESIMD/oneDNN kernels
```

The bundled runtime preserves the accepted standalone history through
`nunchaku-torch@f1dffcd`. The Docker image pins one custom-node revision plus
the reviewed Comfy Kitchen revision and builds `omni_xpu_kernel` for the actual
XPU target. A native wheel from another target is not interchangeable.

## Supported Workflows

All sample images generated on **Intel Arc B580** (11 GB VRAM), 512x512, prompt: *"a cute cat sitting on a windowsill, highly detailed, 4k photography"*.

| Z-Image Turbo (9 steps) | FLUX.1-schnell (4 steps) | Qwen-Image Lightning (4 steps) |
|:---:|:---:|:---:|
| ![Z-Image](images/zimage_xpu_sample.png) | ![FLUX](images/flux_xpu_sample.png) | ![QwenImage](images/qwenimage_xpu_sample.png) |

## Requirements

- Intel Arc GPU (A-series or B-series) with XPU support
- PyTorch with XPU backend (`torch.xpu`)
- [Comfy Kitchen XPU](https://github.com/xiangyuT/comfy-kitchen-xpu) with the
  `svdquant_w4a16_linear` XPU capability
- [`omni_xpu_kernel`](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel)
  with ESIMD/oneDNN INT4 GEMM kernels

At startup this custom node logs the Kitchen version, XPU backend state, and
SVDQuant W4A16 capability. Missing Kitchen/native support falls back to the
more precise W4A4 path; it must not silently reintroduce direct native calls in
the ComfyUI adapter.

## Text Encoders

| Model | Text Encoder | Loader Node |
|-------|-------------|-------------|
| Z-Image Turbo | `Qwen3-4B-Q4_K_M.gguf` | CLIPLoaderGGUF, type=lumina2 |
| FLUX.1-schnell | `clip_l.safetensors` + `t5-v1_1-xxl-encoder-Q4_K_M.gguf` | DualCLIPLoaderGGUF, type=flux |
| Qwen-Image | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | CLIPLoader, type=qwen_image |

## VAE

| Model | VAE File |
|-------|----------|
| Z-Image / FLUX | `ae.safetensors` |
| Qwen-Image | `qwen_image_vae.safetensors` ([download](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors)) |
