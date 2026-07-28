# nunchaku-torch

Standalone PyTorch runtime for Nunchaku SVDQuant W4A4 inference on CPU, XPU, and CUDA.

## Overview

- Pure Python package — no custom C/CUDA extensions required
- Supports CPU, Intel XPU (via Comfy Kitchen and
  `omni_xpu_kernel` ESIMD/oneDNN kernels), and CUDA backends
- SVDQuant W4A4 quantized model inference with LoRA support
- Currently validated models: ZImage (Z-Image-Turbo), QwenImage

## Install

```bash
# Install PyTorch for your backend first, then:
pip install -e .

# For XPU acceleration, install the reviewed Comfy Kitchen integration and
# the target-specific omni_xpu_kernel companion wheel:
pip install --no-deps \
  "git+https://github.com/xiangyuT/comfy-kitchen-xpu.git@399afcc"
cd /path/to/omni_xpu_kernel && pip install -e . --no-build-isolation
```

## CLI

```bash
nunchaku-torch \
  --quant-path /path/to/model.safetensors \
  --base-model /path/to/Z-Image-Turbo \
  --prompt "a realistic photo of a red bicycle" \
  --device auto \
  --height 1024 --width 1024 \
  --steps 9 \
  --output out.png
```

## Python API

```python
from nunchaku_torch import GenerationConfig, generate_image

config = GenerationConfig(
    quant_path="/path/to/model.safetensors",
    base_model="/path/to/Z-Image-Turbo",
    prompt="a tiny pixel-art cat",
    device="xpu",
    height=1024, width=1024,
    steps=9,
)

image = generate_image(config)
image.save("out.png")
```

## Package Structure

```
nunchaku_torch/
  __init__.py          # Lazy-loading public API
  device.py            # Device resolution (cpu/xpu/cuda)
  zimage.py            # ZImage pipeline (GenerationConfig, load_pipeline, generate_image)
  zimage_cli.py        # CLI entry point
  ops/                 # Backend-dispatched ops (CPU fallback, XPU ESIMD/oneDNN, CUDA)
  models/
    linear.py          # SVDQW4A4Linear, AWQW4A16Linear
    attention.py       # Quantized attention and feedforward wrappers
    embeddings.py      # Rotary embeddings
    transformers/      # Model-specific transformer implementations
    attention_processors/  # Model-specific attention processors
    unets/             # UNet implementations (SDXL)
  lora/flux/           # LoRA weight packing utilities
```

## Benchmarking

```bash
# E2E benchmark on XPU
python scripts/benchmark.py --backend runtime --device xpu \
  --quant-path /path/to/model.safetensors \
  --base-model /path/to/Z-Image-Turbo \
  --height 1024 --width 1024 --steps 9

# Validation smoke test
python scripts/validate.py --device xpu \
  --quant-path /path/to/model.safetensors \
  --base-model /path/to/Z-Image-Turbo
```

## Notes

- Self-contained: does not depend on the `nunchaku` PyPI package
- The default XPU SVDQuant W4A16 route is managed by Comfy Kitchen. Kitchen
  owns capability detection, preparation, dispatch, diagnostics, and safe
  fallback; `nunchaku-torch` retains the model and weight lifecycle.
- XPU acceleration requires `omni_xpu_kernel` from
  [Intel llm-scaler](https://github.com/intel/llm-scaler) with ESIMD RMSNorm,
  Rotary, and oneDNN INT4 GEMM kernels.
- For multi-GPU setups, use `ZE_AFFINITY_MASK` to select XPU devices
