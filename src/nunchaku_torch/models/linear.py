import torch
from torch import nn

from ..ops.gemm import svdq_gemm_w4a4_cuda
from ..ops.gemv import awq_gemv_w4a16_cuda
from ..ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda



class SVDQW4A4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super(SVDQW4A4Linear, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.precision = precision
        self.torch_dtype = torch_dtype

        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        else:
            raise ValueError(f"Invalid precision: {precision}")

        self.qweight = nn.Parameter(
            torch.empty(
                out_features, in_features // 2, dtype=torch.int8, device=device
            ),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch_dtype, device=device),
                requires_grad=True,
            )
            if bias
            else None
        )

        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )

        self.proj_down = nn.Parameter(
            torch.empty(in_features, rank, dtype=torch_dtype, device=device)
        )
        self.proj_up = nn.Parameter(
            torch.empty(out_features, rank, dtype=torch_dtype, device=device)
        )

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(
                torch.ones(out_features, dtype=torch_dtype, device=device),
                requires_grad=False,
            )
            self.wtscale = 1.0
        else:
            self.wtscale = None
            self.wcscales = None

        self.act_unsigned = act_unsigned

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        in_features = kwargs.pop("in_features", linear.in_features)
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        device = kwargs.pop("device", linear.weight.device)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )

    def forward(
        self, x: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        x = x.reshape(batch_size * seq_len, channels)

        # XPU: direct W4A16 GEMM (skip activation quantization for speed)
        if self.qweight.device.type == "xpu":
            result = self._forward_xpu(x)
            return result.reshape(batch_size, seq_len, -1)

        if output is None:
            output = torch.empty(
                batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device
            )
        quantized_x, ascales, lora_act_out = self.quantize(x)
        output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
        output = output.reshape(batch_size, seq_len, -1)
        return output

    def _forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        """W4A16 or W4A4 GEMM on XPU, auto-selected per model.

        W4A16 (fast): fused smooth → fp16 → oneDNN INT4 GEMM. Works for Z-Image/FLUX.
        W4A4 (precise): quantize → dequant → fp32 matmul. Required for QwenImage
        where W4A16 precision is insufficient (10% cosine error per layer).

        Selection: W4A4 is used when act_unsigned flag is set (fused_gelu_mlp chain)
        or can be forced via _xpu_force_w4a4 attribute.
        """
        try:
            from omni_xpu_kernel import svdq as omni_svdq
        except ImportError:
            M = x.shape[0]
            output = torch.empty(M, self.out_features, dtype=x.dtype, device=x.device)
            quantized_x, ascales, lora_act = self.quantize(x)
            self.forward_quant(quantized_x, ascales, lora_act, output)
            return output

        # Force W4A4 for layers that need it (set by model wrapper)
        if getattr(self, '_xpu_force_w4a4', False):
            M = x.shape[0]
            output = torch.empty(M, self.out_features, dtype=x.dtype, device=x.device)
            quantized_x, ascales, lora_act = self.quantize(x)
            self.forward_quant(quantized_x, ascales, lora_act, output)
            return output

        # W4A16 fast path
        input_device = x.device
        x_orig_dtype = x.dtype
        xpu_device = self.qweight.device

        if x.device != xpu_device:
            x = x.to(xpu_device)

        # Apply smooth factor: fused ESIMD kernel (fp16 output, zero extra allocation).
        # nan_to_num_ replaces rare inf from fp16 overflow in-place.
        if self.smooth_factor is not None:
            if not hasattr(self, '_xpu_rcp_smooth') or self._xpu_rcp_smooth is None:
                self._xpu_rcp_smooth = (1.0 / self.smooth_factor.float()).to(torch.float16)
            x_gemm = omni_svdq.fused_smooth_mul_convert(x, self._xpu_rcp_smooth)
            x_gemm.nan_to_num_(nan=0.0, posinf=65504.0, neginf=-65504.0)
        else:
            x_gemm = x.to(torch.float16)

        # Prepare oneDNN weights (cached)
        if not hasattr(self, '_xpu_packed_u4') or self._xpu_packed_u4 is None:
            wgt_u8 = self.qweight.view(torch.uint8)
            self._xpu_packed_u4, self._xpu_scales_f16 = omni_svdq.prepare_onednn_weights(
                wgt_u8, self.wscales
            )

        # W4A16 GEMM with fused LoRA+bias
        has_lora = self.proj_down is not None and self.proj_up is not None
        if has_lora or self.bias is not None:
            M, N = x_gemm.shape[0], self.out_features
            dst = torch.zeros(M, N, dtype=torch.bfloat16, device=x_gemm.device)
            if has_lora:
                lora_out = (x.to(torch.bfloat16) @ self.proj_down.to(torch.bfloat16)) @ self.proj_up.to(torch.bfloat16).T
                dst.add_(lora_out)
                del lora_out
            if self.bias is not None:
                dst.add_(self.bias.to(torch.bfloat16))
            omni_svdq.onednn_int4_gemm_add_to_output(
                x_gemm, self._xpu_packed_u4, self._xpu_scales_f16, dst
            )
            result = dst
        else:
            result = omni_svdq.onednn_int4_gemm_preconverted(
                x_gemm, self._xpu_packed_u4, self._xpu_scales_f16
            )

        return result.to(x_orig_dtype).to(input_device)

    def quantize(
        self, x: torch.Tensor, pad_size: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self.proj_down,
            smooth=self.smooth_factor,
            fp4=self.precision == "nvfp4",
            pad_size=pad_size,
        )
        return quantized_x, ascales, lora_act_out

    def forward_quant(
        self,
        quantized_x: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        provided_output = output is not None
        if output is None:
            output_dtype = self.proj_up.dtype
            if quantized_x.device.type == "cpu":
                output_dtype = torch.float32
            output = torch.empty(
                quantized_x.shape[0],
                self.out_features,
                dtype=output_dtype,
                device=quantized_x.device,
            )

        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            bias=self.bias,
            fp4=self.precision == "nvfp4",
            alpha=self.wtscale,
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
        )
        if quantized_x.device.type == "cpu" and not provided_output:
            return output
        return output

    def __repr__(self):
        return (
            f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"
        )


class AWQW4A16Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super(AWQW4A16Linear, self).__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(
                out_features // 4, in_features // 2, dtype=torch.int32, device=device
            ),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch_dtype, device=device),
                requires_grad=True,
            )
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.wzeros = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype,
                device=device,
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_device = x.device
        weight_device = self.qweight.device

        # Handle CPU input + XPU weights
        if input_device != weight_device:
            x = x.to(weight_device)

        output = awq_gemv_w4a16_cuda(
            in_feats=x,
            kernel=self.qweight,
            scaling_factors=self.wscales,
            zeros=self.wzeros,
            m=x.shape[0],
            n=self.out_features,
            k=self.in_features,
            group_size=self.group_size,
        )
        if self.bias is not None:
            view_shape = [1] * (output.ndim - 1) + [-1]
            output.add_(self.bias.view(view_shape))

        if output.device != input_device:
            output = output.to(input_device)
        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        **kwargs,
    ):
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            torch_dtype=torch_dtype,
            device=device,
        )

    def __repr__(self):
        return f"AWQW4A16Linear(in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size})"
