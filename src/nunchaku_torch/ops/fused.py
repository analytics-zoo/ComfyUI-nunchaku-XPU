import torch
from diffusers.models.normalization import RMSNorm as DiffUsersRMSNorm
from torch.nn import RMSNorm

from ..models.linear import SVDQW4A4Linear
from ..utils import ceil_divide
from .gemm import svdq_gemm_w4a4_cuda


def fused_gelu_mlp(
    x: torch.Tensor, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear, pad_size: int = 256
) -> torch.Tensor:
    # XPU: default W4A4 fused path (safe for all models).
    # W4A16 fast path available via _xpu_use_w4a16_fused flag, but needs extra
    # memory for bf16 GELU intermediate which can OOM on constrained GPUs.
    if fc1.qweight.device.type == "xpu":
        if getattr(fc1, '_xpu_use_w4a16_fused', False):
            return _fused_gelu_mlp_xpu(x, fc1, fc2)
        return _fused_gelu_mlp_xpu_w4a4(x, fc1, fc2)

    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = fc1.quantize(x)

    batch_size_pad = ceil_divide(batch_size * seq_len, pad_size) * pad_size

    qout_act = torch.empty(
        batch_size_pad, fc1.out_features // 2, dtype=torch.uint8, device=x.device
    )
    if fc2.precision == "nvfp4":
        qout_ascales = torch.empty(
            fc1.out_features // 16,
            batch_size_pad,
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
    else:
        qout_ascales = torch.empty(
            fc1.out_features // 64, batch_size_pad, dtype=x.dtype, device=x.device
        )
    qout_lora_act = torch.empty(
        batch_size_pad, fc2.proj_down.shape[1], dtype=torch.float32, device=x.device
    )

    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=fc1.qweight,
        qout=qout_act,
        ascales=ascales,
        wscales=fc1.wscales,
        oscales=qout_ascales,
        lora_act_in=lora_act,
        lora_up=fc1.proj_up,
        lora_down=fc2.proj_down,
        lora_act_out=qout_lora_act,
        bias=fc1.bias,
        smooth_factor=fc2.smooth_factor,
        fp4=fc1.precision == "nvfp4",
        alpha=fc1.wtscale,
        wcscales=fc1.wcscales,
    )
    output_dtype = x.dtype
    if x.device.type == "cpu":
        output_dtype = torch.float32
    output = torch.empty(
        batch_size * seq_len, fc2.out_features, dtype=output_dtype, device=x.device
    )
    output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act, output=output)
    output = output.view(batch_size, seq_len, -1)
    # Convert back to input dtype if we used float32 for CPU accumulation
    if output.dtype != x.dtype:
        output = output.to(x.dtype)
    return output


def _fused_gelu_mlp_xpu(
    x: torch.Tensor, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear,
) -> torch.Tensor:
    """XPU fused GELU MLP using W4A16 oneDNN GEMM for both fc1 and fc2.

    Decomposed path (no INT4 activation quantization):
      1. fc1: W4A16 GEMM (smooth + oneDNN INT4 GEMM + LoRA + bias)
      2. GELU activation
      3. fc2: apply smooth(fc2) + W4A16 GEMM + LoRA
    """
    from omni_xpu_kernel import svdq as omni_svdq
    import torch.nn.functional as F

    batch_size, seq_len, channels = x.shape
    M = batch_size * seq_len
    x_2d = x.view(M, channels)

    # Step 1: fc1 W4A16
    fc1_out = fc1._forward_xpu(x_2d)  # [M, fc1.out_features]

    # Step 2: GELU + smooth(fc2) → fp16 for oneDNN GEMM
    # Fuse GELU + smooth + convert in minimal memory: avoid keeping both
    # fc1_out and fc1_gelu alive simultaneously.
    fc1_out = F.gelu(fc1_out, approximate="tanh")  # in-place reuse

    # LoRA down for fc2 must use GELU output (before smooth)
    has_lora = fc2.proj_down is not None and fc2.proj_up is not None
    if has_lora:
        lora_out = (fc1_out.float() @ fc2.proj_down.float()) @ fc2.proj_up.float().T

    if fc2.smooth_factor is not None:
        if not hasattr(fc2, '_xpu_rcp_smooth') or fc2._xpu_rcp_smooth is None:
            fc2._xpu_rcp_smooth = (1.0 / fc2.smooth_factor.float()).to(torch.float16)
        x_gemm = omni_svdq.fused_smooth_mul_convert(fc1_out, fc2._xpu_rcp_smooth)
        x_gemm.nan_to_num_(nan=0.0, posinf=65504.0, neginf=-65504.0)
    else:
        x_gemm = fc1_out.to(torch.float16)
    del fc1_out  # free large intermediate

    if not hasattr(fc2, '_xpu_packed_u4') or fc2._xpu_packed_u4 is None:
        fc2._xpu_packed_u4, fc2._xpu_scales_f16 = omni_svdq.prepare_onednn_weights(
            fc2.qweight.view(torch.uint8), fc2.wscales
        )

    # fc2 GEMM + LoRA + bias
    if has_lora or fc2.bias is not None:
        N = fc2.out_features
        dst = torch.zeros(M, N, dtype=torch.bfloat16, device=x.device)
        if has_lora:
            dst.add_(lora_out.to(torch.bfloat16))
            del lora_out
        if fc2.bias is not None:
            dst.add_(fc2.bias.to(torch.bfloat16))
        omni_svdq.onednn_int4_gemm_add_to_output(
            x_gemm, fc2._xpu_packed_u4, fc2._xpu_scales_f16, dst
        )
        result = dst
    else:
        result = omni_svdq.onednn_int4_gemm_preconverted(
            x_gemm, fc2._xpu_packed_u4, fc2._xpu_scales_f16
        )

    return result.to(x.dtype).view(batch_size, seq_len, -1)


def _fused_gelu_mlp_xpu_w4a4(
    x: torch.Tensor, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear,
) -> torch.Tensor:
    """W4A4 precise path for models that need it (e.g. QwenImage)."""
    batch_size, seq_len, channels = x.shape
    x_2d = x.view(batch_size * seq_len, channels)

    quantized_x, ascales, lora_act = fc1.quantize(x_2d)
    batch_size_pad = quantized_x.shape[0]

    qout_act = torch.empty(
        batch_size_pad, fc1.out_features // 2, dtype=torch.uint8, device=x.device
    )
    qout_ascales = torch.empty(
        fc1.out_features // 64, batch_size_pad, dtype=x.dtype, device=x.device
    )
    qout_lora_act = torch.empty(
        batch_size_pad, fc2.proj_down.shape[1], dtype=torch.float32, device=x.device
    )

    svdq_gemm_w4a4_cuda(
        act=quantized_x, wgt=fc1.qweight, qout=qout_act,
        ascales=ascales, wscales=fc1.wscales, oscales=qout_ascales,
        lora_act_in=lora_act, lora_up=fc1.proj_up,
        lora_down=fc2.proj_down, lora_act_out=qout_lora_act,
        bias=fc1.bias, smooth_factor=fc2.smooth_factor,
        fp4=fc1.precision == "nvfp4", alpha=fc1.wtscale, wcscales=fc1.wcscales,
    )

    output = torch.empty(
        batch_size * seq_len, fc2.out_features, dtype=x.dtype, device=x.device
    )
    output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act, output=output)
    return output.view(batch_size, seq_len, -1)


def fused_qkv_norm_rottary(
    x: torch.Tensor,
    proj: SVDQW4A4Linear,
    norm_q: RMSNorm | None = None,
    norm_k: RMSNorm | None = None,
    rotary_emb: torch.Tensor | None = None,
    output: torch.Tensor
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    attn_tokens: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (
        norm_q is None
        or isinstance(norm_q, RMSNorm)
        or (isinstance(norm_q, DiffUsersRMSNorm) and norm_q.bias is None)
    )
    assert (
        norm_k is None
        or isinstance(norm_k, RMSNorm)
        or (isinstance(norm_k, DiffUsersRMSNorm) and norm_k.bias is None)
    )

    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = proj.quantize(x)

    if output is None:
        output = torch.empty(
            batch_size * seq_len, proj.out_features, dtype=x.dtype, device=x.device
        )

    if isinstance(output, tuple):
        assert len(output) == 3
        output_q, output_k, output_v = output
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=proj.qweight,
            ascales=ascales,
            wscales=proj.wscales,
            lora_act_in=lora_act,
            lora_up=proj.proj_up,
            bias=proj.bias,
            fp4=proj.precision == "nvfp4",
            alpha=proj.wtscale,
            wcscales=proj.wcscales,
            norm_q=norm_q.weight if norm_q is not None else None,
            norm_k=norm_k.weight if norm_k is not None else None,
            rotary_emb=rotary_emb,
            out_q=output_q,
            out_k=output_k,
            out_v=output_v,
            attn_tokens=attn_tokens,
        )
        return output_q, output_k, output_v
    else:
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=proj.qweight,
            out=output,
            ascales=ascales,
            wscales=proj.wscales,
            lora_act_in=lora_act,
            lora_up=proj.proj_up,
            bias=proj.bias,
            fp4=proj.precision == "nvfp4",
            alpha=proj.wtscale,
            wcscales=proj.wcscales,
            norm_q=norm_q.weight if norm_q is not None else None,
            norm_k=norm_k.weight if norm_k is not None else None,
            rotary_emb=rotary_emb,
        )
        output = output.view(batch_size, seq_len, -1)
        return output
