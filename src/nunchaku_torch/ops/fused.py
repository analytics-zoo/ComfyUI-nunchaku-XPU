import torch
from diffusers.models.normalization import RMSNorm as DiffUsersRMSNorm
from torch.nn import RMSNorm

from ..models.linear import SVDQW4A4Linear
from ..utils import ceil_divide
from .gemm import svdq_gemm_w4a4_cuda


def fused_gelu_mlp(
    x: torch.Tensor, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear, pad_size: int = 256
) -> torch.Tensor:
    # XPU: use fused path that preserves quantization chain
    if fc1.qweight.device.type == "xpu":
        return _fused_gelu_mlp_xpu(x, fc1, fc2)

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
    """XPU fused GELU MLP matching CPU fused semantics.

    Replicates the CPU fused path:
      1. fc1: quantize input → W4A4 GEMM (via XPU dispatch)
      2. GELU + shift(+0.171875) + smooth(fc2) + quantize to unsigned INT4
      3. fc2: quantized GEMM with unsigned INT4 input

    This preserves the same quantization behavior as the CPU fused path,
    ensuring image quality matches.
    """
    batch_size, seq_len, channels = x.shape
    x_2d = x.view(batch_size * seq_len, channels)

    # Step 1: fc1 quantize + GEMM (uses xpu_ops dispatch automatically)
    quantized_x, ascales, lora_act = fc1.quantize(x_2d)

    batch_size_pad = quantized_x.shape[0]

    # Allocate qout buffers for fused GELU+quantize
    qout_act = torch.empty(
        batch_size_pad, fc1.out_features // 2, dtype=torch.uint8, device=x.device
    )
    qout_ascales = torch.empty(
        fc1.out_features // 64, batch_size_pad, dtype=x.dtype, device=x.device
    )
    qout_lora_act = torch.empty(
        batch_size_pad, fc2.proj_down.shape[1], dtype=torch.float32, device=x.device
    )

    # Step 2: fc1 GEMM with qout (fuses GELU + quantize for fc2)
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

    # Step 3: fc2 quantized GEMM
    output_dtype = x.dtype
    if x.device.type == "cpu":
        output_dtype = torch.float32
    output = torch.empty(
        batch_size * seq_len, fc2.out_features, dtype=output_dtype, device=x.device
    )
    output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act, output=output)
    output = output.view(batch_size, seq_len, -1)
    if output.dtype != x.dtype:
        output = output.to(x.dtype)
    return output


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
