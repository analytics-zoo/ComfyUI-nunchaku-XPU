"""
XPU ops backend for nunchaku_torch.

Implements the 3 core nunchaku ops interfaces using omni_xpu_kernel's atomic kernels:
1. svdq_quantize_w4a4_act_fuse_lora_xpu  (smooth + quantize_act + lora_down matmul)
2. svdq_gemm_w4a4_xpu                    (onednn_int4_gemm + lora_up + bias + silu)
3. awq_gemv_w4a16_xpu                    (dequant + matmul)

All functions match the CPU reference signatures in cpu_ops.py for drop-in replacement.
"""

import torch
import torch.nn.functional as F

from ..utils import ceil_divide

# Lazy import omni_xpu_kernel to avoid import errors on non-XPU systems
_omni = None


def _get_omni():
    global _omni
    if _omni is None:
        import omni_xpu_kernel
        _omni = omni_xpu_kernel
    return _omni


# ============================================================================
# Op 1: svdq_quantize_w4a4_act_fuse_lora_xpu
# ============================================================================

def svdq_quantize_w4a4_act_fuse_lora_xpu(
    input: torch.Tensor,
    output: torch.Tensor,
    oscales: torch.Tensor,
    lora_down: torch.Tensor | None,
    lora_act_out: torch.Tensor | None,
    smooth: torch.Tensor | None,
    fuse_glu: bool = False,
    fp4: bool = False,
) -> None:
    """XPU implementation of activation quantization with LoRA fusion.

    Composes omni_xpu_kernel primitives:
      1. fused_smooth_mul_convert (smooth + bf16->f16 conversion)
      2. quantize_act_int4 (per-group INT4 quantization)
      3. torch.matmul (LoRA down projection)
    """
    if fp4:
        raise NotImplementedError("XPU backend does not support NVFP4 (fp4)")
    if fuse_glu:
        raise NotImplementedError("XPU backend does not support fused GLU")

    omni = _get_omni()
    M, K = input.shape
    M_pad = output.shape[0]

    # Resolve device: use XPU if smooth is on XPU, else CPU
    compute_device = smooth.device if smooth is not None else input.device
    if compute_device.type != "xpu" and lora_down is not None:
        compute_device = lora_down.device

    # Move input to compute device if needed
    inp = input.to(compute_device) if input.device != compute_device else input

    # Step 1: LoRA down projection (on original input, before smoothing)
    if lora_down is not None and lora_act_out is not None:
        lora_result = inp.float() @ lora_down.to(compute_device).float()
        lora_act_out[:M].copy_(lora_result.to(lora_act_out.dtype).to(lora_act_out.device))
        if M_pad > M:
            lora_act_out[M:].zero_()

    # Step 2: Apply smoothing in fp32 — use division (not multiply-by-reciprocal)
    # to match CPU reference exactly. IEEE 754: a/b != a*(1/b) due to intermediate rounding.
    if smooth is not None:
        smooth_dev = smooth.to(compute_device).float()
        x_smooth = inp.float() / smooth_dev
    else:
        x_smooth = inp.float()

    # Step 3: Pad to M_pad
    if M_pad > M:
        x_padded = torch.zeros(M_pad, K, dtype=x_smooth.dtype, device=compute_device)
        x_padded[:M] = x_smooth
    else:
        x_padded = x_smooth.contiguous()

    # Step 4: Quantize activation to INT4
    packed_act, ascales = omni.svdq.quantize_act_int4(x_padded, group_size=64)

    # Step 5: Copy to output buffers (handle device mismatch)
    packed_out = packed_act.to(output.device)
    if output.dtype != packed_out.dtype:
        output.copy_(packed_out.view(output.dtype))
    else:
        output.copy_(packed_out)
    oscales.copy_(ascales.to(oscales.dtype).to(oscales.device))


# ============================================================================
# Op 2: svdq_gemm_w4a4_xpu
# ============================================================================

# Cache for pre-converted oneDNN weights: {data_ptr: (packed_u4, scales_f16)}
_weight_cache = {}


def _get_prepared_weights(wgt: torch.Tensor, wscales: torch.Tensor):
    """Get or create pre-converted oneDNN weights (cached by data_ptr)."""
    key = wgt.data_ptr()
    if key not in _weight_cache:
        omni = _get_omni()
        wgt_u8 = wgt.view(torch.uint8)
        packed_u4, scales_f16 = omni.svdq.prepare_onednn_weights(wgt_u8, wscales)
        _weight_cache[key] = (packed_u4, scales_f16)
    return _weight_cache[key]


def svdq_gemm_w4a4_xpu(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    poolout: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    out_vk: torch.Tensor | None = None,
    out_linearattn: torch.Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
    lora_mode: str = "naive",
    lora_up_effective: torch.Tensor | None = None,
    bias_effective: torch.Tensor | None = None,
) -> None:
    """XPU implementation of quantized GEMM with LoRA fusion.

    Composes omni_xpu_kernel primitives:
      1. dequantize_w4 (unpack INT4 activations back to f16)
      2. onednn_int4_gemm_preconverted (W4 GEMM via oneDNN)
      3. torch.matmul (LoRA residual: lora_act @ lora_up.T)
      4. torch.add (bias)
      5. F.silu (optional fuse_silu for SwiGLU)

    For qout path (fused_gelu_mlp): also handles GELU + next-layer quantization.
    """
    if fp4:
        raise NotImplementedError("XPU backend does not support NVFP4 (fp4)")
    if ascales is None or wscales is None:
        raise ValueError("ascales and wscales are required")

    omni = _get_omni()
    if alpha is None:
        alpha = 1.0

    M = act.shape[0]
    N = wgt.shape[0]

    # Resolve compute device
    compute_device = wgt.device if wgt.device.type == "xpu" else act.device

    # Move tensors to compute device
    act_dev = act.to(compute_device) if act.device != compute_device else act
    wgt_dev = wgt.to(compute_device) if wgt.device != compute_device else wgt
    ascales_dev = ascales.to(compute_device) if ascales.device != compute_device else ascales
    wscales_dev = wscales.to(compute_device) if wscales.device != compute_device else wscales

    # W4A4 GEMM: unpack INT4 → dequant with per-group scales → fp32 matmul.
    # Vectorized dequant (no Python loop) + single large fp32 matmul.
    # fp32 is exact for INT4*INT4 products (max 8*8=64, well within fp32 precision).
    # This is 9x faster than the group-wise torch._int_mm loop while matching its precision.
    K = act_dev.shape[1] * 2  # packed INT4: K/2 bytes
    group_size = K // ascales_dev.shape[0]
    num_groups = K // group_size

    # Unpack INT4 to int8
    act_i8 = omni.svdq.unpack_int4(act_dev.view(torch.uint8), True)
    if act_unsigned:
        act_i8 = (act_i8.to(torch.int16) + 8).to(torch.int8)
    wgt_i8 = omni.svdq.unpack_int4(wgt_dev.view(torch.uint8), True)

    # Vectorized dequant: reshape to [M, groups, gs], multiply by per-group scale, flatten
    ascales_f = ascales_dev.to(torch.float32)  # [groups, M]
    wscales_f = wscales_dev.to(torch.float32)  # [groups, N]

    act_deq = act_i8.float().reshape(M, num_groups, group_size)
    act_deq *= ascales_f.T.unsqueeze(2)  # [M, groups] → [M, groups, 1]
    wgt_deq = wgt_i8.float().reshape(N, num_groups, group_size)
    wgt_deq *= wscales_f.T.unsqueeze(2)  # [N, groups] → [N, groups, 1]

    result = (act_deq.reshape(M, K) @ wgt_deq.reshape(N, K).T).to(torch.bfloat16)
    del act_i8, wgt_i8, act_deq, wgt_deq

    # Step 3: Apply alpha scaling
    if alpha != 1.0:
        result = result * alpha

    # Step 4: Apply per-channel scales (wcscales)
    if wcscales is not None:
        result = result * wcscales.to(result.dtype).view(1, -1)

    # Step 5: LoRA residual (move to compute device if needed)
    if lora_act_in is not None:
        lora_act_dev = lora_act_in.to(compute_device) if lora_act_in.device != compute_device else lora_act_in
        lora_up_dev = lora_up.to(compute_device) if lora_up is not None and lora_up.device != compute_device else lora_up
        bias_dev = bias.to(compute_device) if bias is not None and bias.device != compute_device else bias
        lora_up_eff_dev = lora_up_effective.to(compute_device) if lora_up_effective is not None and lora_up_effective.device != compute_device else lora_up_effective
        bias_eff_dev = bias_effective.to(compute_device) if bias_effective is not None and bias_effective.device != compute_device else bias_effective
        lora_contrib = _compute_lora_residual_xpu(
            lora_act_dev, lora_up_dev, bias_dev, lora_mode,
            lora_up_eff_dev, bias_eff_dev, lora_scales,
        )
        result = result + lora_contrib.to(result.dtype)
    elif bias is not None:
        result = result + bias.to(compute_device).to(result.dtype).view(1, -1)

    # Step 6: Fuse SiLU (for SwiGLU in fused_gelu_mlp path)
    if fuse_silu:
        result = result * torch.sigmoid(result)

    # Step 7: Write to output buffer(s)
    if out is not None:
        M_out = out.shape[0]
        N_out = out.shape[1]
        out.copy_(result[:M_out, :N_out].to(out.dtype).to(out.device))

    # Step 8: Handle qout path (fused_gelu_mlp: quantize output for next layer)
    if qout is not None and oscales is not None:
        _quantize_output_for_next_layer_xpu(
            result, qout, oscales, lora_down, lora_act_out, smooth_factor,
        )


def _compute_lora_residual_xpu(
    lora_act_in, lora_up, bias, lora_mode,
    lora_up_effective, bias_effective, lora_scales,
):
    """Compute LoRA residual: lora_act @ lora_up.T + bias."""
    lora_act = lora_act_in.float()

    # Apply per-group lora_scales
    if lora_scales is not None and len(lora_scales) > 0:
        rank = lora_act.shape[1]
        scale_t = torch.ones(rank, dtype=torch.float32, device=lora_act.device)
        for g, s in enumerate(lora_scales):
            start = g * 16
            end = min(start + 16, rank)
            if start >= rank:
                break
            scale_t[start:end] = float(s)
        lora_act = lora_act * scale_t.view(1, -1)

    if lora_mode == "naive":
        if lora_up is None:
            raise ValueError("lora_up is required for lora_mode='naive'")
        residual = lora_act @ lora_up.float().T
        if bias is not None:
            residual = residual + bias.float().view(1, -1)
        return residual

    if lora_mode == "effective_linear":
        if lora_up_effective is None:
            raise ValueError("lora_up_effective required for effective_linear mode")
        residual = lora_act @ lora_up_effective.float()
        if bias_effective is not None:
            residual = residual + bias_effective.float().view(1, -1)
        return residual

    raise ValueError(f"Unsupported lora_mode: {lora_mode}")


def _quantize_output_for_next_layer_xpu(
    result, qout, oscales, lora_down, lora_act_out, smooth_factor,
):
    """GELU + quantize output for fused_gelu_mlp's second linear."""
    omni = _get_omni()
    M, N = result.shape
    M_pad = qout.shape[0]
    shift_gelu = 0.171875

    # Ensure all computation on same device as result
    compute_device = result.device

    # GELU activation
    x_gelu = F.gelu(result.float(), approximate="tanh")

    # LoRA down projection for next layer
    if lora_down is not None and lora_act_out is not None:
        ld = lora_down.to(compute_device) if lora_down.device != compute_device else lora_down
        lora_result = (x_gelu @ ld.float())[:M]
        lora_act_out[:M].copy_(lora_result.to(lora_act_out.dtype).to(lora_act_out.device))
        if M_pad > M:
            lora_act_out[M:].zero_()

    # Add shift for unsigned quantization
    x = x_gelu + shift_gelu

    # Apply smooth factor for next layer (bf16 to avoid fp16 overflow, nan_to_num for inf)
    if smooth_factor is not None:
        sf = smooth_factor.to(compute_device) if smooth_factor.device != compute_device else smooth_factor
        rcp_smooth = (1.0 / sf.float()).to(torch.bfloat16)
        x_f16 = (x.to(torch.bfloat16) * rcp_smooth).to(torch.float16)
        x_f16.nan_to_num_(nan=0.0, posinf=65504.0, neginf=-65504.0)
    else:
        x_f16 = x.to(torch.float16)

    # Pad
    if M_pad > M:
        x_padded = torch.zeros(M_pad, N, dtype=x_f16.dtype, device=compute_device)
        x_padded[:M] = x_f16
    else:
        x_padded = x_f16.contiguous()

    # Quantize to unsigned INT4 [0, 15] for fc2's act_unsigned path.
    # omni quantize_act_int4 always produces signed [-8, 7]. We quantize as signed
    # and the GEMM unpack adds +8 to recover unsigned values. This is mathematically
    # equivalent: stored_signed = value_unsigned - 8, and GEMM does +8 to recover.
    # But the scale must be computed for unsigned range [0, 15] not signed [-8, 7].
    # Since input is GELU+shift (all non-negative), signed quantize uses only [0, 7]
    # wasting half the range. Instead: compute unsigned scale, quantize, then store
    # as signed (subtract 8) so GEMM's +8 recovers the correct unsigned value.
    gs = 64
    x_grouped = x_padded.float().reshape(M_pad, N // gs, gs)
    # Unsigned scale: max / 15 (input is non-negative after GELU+shift)
    group_max = x_grouped.amax(dim=-1).clamp(min=1e-10)  # [M_pad, N//gs]
    uscale = group_max / 15.0
    # Quantize to unsigned [0, 15] then shift to signed [-8, 7] for storage
    x_quant = (x_grouped / uscale.unsqueeze(-1)).round().clamp(0, 15).to(torch.int8) - 8
    # Pack signed INT4 pairs into uint8
    x_flat = x_quant.reshape(M_pad, N)
    low = x_flat[:, 0::2].to(torch.uint8) & 0x0F
    high = (x_flat[:, 1::2].to(torch.uint8) & 0x0F) << 4
    packed = (low | high).to(torch.uint8)

    qout.copy_(packed.to(qout.device).view(qout.dtype))
    oscales.copy_(uscale.T.to(oscales.dtype).to(oscales.device))


# ============================================================================
# Op 3: awq_gemv_w4a16_xpu
# ============================================================================

def awq_gemv_w4a16_xpu(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    """XPU implementation of AWQ W4A16 GEMV.

    AWQ dequantize + standard matmul. The AWQ packed format uses int16 tiles
    with a specific permutation pattern (from nunchaku's cpu_ops.py).
    """
    weight = _awq_dequantize_xpu(kernel, scaling_factors, zeros, n, k, group_size)
    x = in_feats.float().reshape(m, k)
    output = x @ weight.float().T
    return output.to(in_feats.dtype)


def _awq_dequantize_xpu(
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
) -> torch.Tensor:
    """Dequantize AWQ int4 weights to float.

    Mirrors cpu_ops.awq_dequantize_weights exactly for correctness.
    """
    # Unpack using the same algorithm as cpu_ops.awq_unpack_weights
    packed16 = kernel.view(torch.int16).reshape(n // 4, k)
    packed = (
        packed16.reshape(n // 4, k // 64, 4, 16)
        .permute(0, 2, 1, 3)
        .reshape(n, k // 32, 8)
        .to(torch.int32)
    )

    masks = torch.tensor(
        [0xF, 0xF0, 0xF00, 0xF000], dtype=torch.int32, device=packed.device
    )
    shifts = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device=packed.device)
    parts = ((packed.unsqueeze(-1) & masks) >> shifts).to(torch.float32)

    weight = torch.zeros(n, k // 32, 32, dtype=torch.float32, device=packed.device)
    for idx in range(8):
        weight[:, :, idx] = parts[:, :, idx, 0]
        weight[:, :, idx + 8] = parts[:, :, idx, 1]
        weight[:, :, idx + 16] = parts[:, :, idx, 2]
        weight[:, :, idx + 24] = parts[:, :, idx, 3]

    weight = weight.reshape(n, k)

    # Dequantize: weight = weight * scale + zero
    num_groups = k // group_size
    weight = weight.view(n, num_groups, group_size)
    sc = scaling_factors.float().T.unsqueeze(-1)
    zp = zeros.float().T.unsqueeze(-1)
    dequant = weight * sc + zp
    return dequant.view(n, k)
