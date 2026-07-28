import json
import os
from pathlib import Path
from typing import List, Optional, cast

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_z_image import (
    FeedForward as ZImageFeedForward,
)
from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
)
from huggingface_hub import utils

from ..attention import NunchakuBaseAttention
from ..attention_processors.zimage import NunchakuZSingleStreamAttnProcessor
from ..embeddings import pack_rotemb
from ..linear import SVDQW4A4Linear
from ..unets.unet_sdxl import NunchakuSDXLFeedForward
from ..utils import fuse_linears
from ...ops.gemm import svdq_gemm_w4a4_cuda
from ...ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from ...utils import get_precision, pad_tensor
from .utils import (
    NunchakuModelLoaderMixin,
    convert_fp16,
    decode_int4_state_dict_for_cpu,
    patch_scale_key,
)


class NunchakuZImageRopeHook:
    def __init__(self):
        self.packed_cache = {}

    def __call__(self, module: nn.Module, input_args: tuple, input_kwargs: dict):
        freqs_cis = input_kwargs.get("freqs_cis", None)
        if not isinstance(freqs_cis, torch.Tensor):
            return None
        cache_key = freqs_cis.data_ptr()
        packed_freqs_cis = self.packed_cache.get(cache_key, None)
        if packed_freqs_cis is None:
            packed_freqs_cis = torch.view_as_real(freqs_cis).unsqueeze(3)
            packed_freqs_cis = torch.flip(packed_freqs_cis, dims=[-1])
            packed_freqs_cis = pack_rotemb(pad_tensor(packed_freqs_cis, 256, 1))
            self.packed_cache[cache_key] = packed_freqs_cis
        new_input_kwargs = input_kwargs.copy()
        new_input_kwargs["freqs_cis"] = packed_freqs_cis
        return input_args, new_input_kwargs


class NunchakuZImageFusedModule(nn.Module):
    def __init__(self, qkv: SVDQW4A4Linear, norm_q: RMSNorm, norm_k: RMSNorm):
        super().__init__()
        for name, param in qkv.named_parameters(prefix="qkv_"):
            setattr(self, name.replace(".", ""), param)
        self.qkv_precision = qkv.precision
        self.qkv_out_features = qkv.out_features
        for name, param in norm_q.named_parameters(prefix="norm_q_"):
            setattr(self, name.replace(".", ""), param)
        for name, param in norm_k.named_parameters(prefix="norm_k_"):
            setattr(self, name.replace(".", ""), param)

    @staticmethod
    def _apply_rmsnorm_cpu(
        x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        if x.device.type == "xpu":
            try:
                from omni_xpu_kernel import norm as omni_norm
                # omni rms_norm needs 2D [batch, hidden_size]
                orig_shape = x.shape
                hidden = orig_shape[-1]
                x_2d = x.reshape(-1, hidden)
                out_2d = omni_norm.rms_norm(weight, x_2d, eps)
                return out_2d.reshape(orig_shape)
            except (ImportError, RuntimeError):
                pass
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps)
        return (out.to(weight.dtype) * weight).to(x.dtype)

    @staticmethod
    def _apply_rotary_emb_cpu(
        x_in: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        # XPU: try omni_xpu_kernel.rotary for ESIMD-optimized rotary
        if x_in.device.type == "xpu":
            try:
                from omni_xpu_kernel import rotary as omni_rotary
                # Convert complex freqs_cis to cos/sin caches
                if freqs_cis.is_complex():
                    freqs_real = torch.view_as_real(freqs_cis)  # [..., head_dim/2, 2]
                else:
                    freqs_real = freqs_cis.float().reshape(*freqs_cis.shape[:-1], -1, 2)
                # freqs_real: [B?, seq_total, head_dim/2, 2] where [...,0]=cos, [...,1]=sin
                # But complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                # freqs_cis = cos + i*sin, so real part = cos, imag part = sin

                # omni rotary expects: x=[total_rows, head_dim], cos=[S, D/2], sin=[S, D/2]
                B, S, H, D = x_in.shape
                seq_len = S

                # Extract cos/sin from complex freqs
                if freqs_cis.is_complex():
                    fc = freqs_cis
                    if fc.shape[-2] > seq_len:
                        fc = fc[..., :seq_len, :]
                    cos_cache = fc.real.float()  # [..., seq, head_dim/2]
                    sin_cache = fc.imag.float()
                else:
                    fc = freqs_real
                    if fc.shape[-3] > seq_len:
                        fc = fc[..., :seq_len, :, :]
                    cos_cache = fc[..., 0].float()
                    sin_cache = fc[..., 1].float()

                # Squeeze batch dim if present
                if cos_cache.ndim > 2:
                    cos_cache = cos_cache.squeeze(0)
                    sin_cache = sin_cache.squeeze(0)
                # cos_cache: [seq, head_dim/2], sin_cache: [seq, head_dim/2]

                # omni rotary: x=[total_rows, head_dim]
                x_flat = x_in.reshape(B * S * H, D)
                out_flat = omni_rotary.rotary_emb(x_flat, cos_cache, sin_cache, seq_len, H)
                return out_flat.reshape(B, S, H, D)
            except (ImportError, RuntimeError):
                pass  # fallback to PyTorch

        device_type = x_in.device.type if x_in.device.type in ("cuda",) else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            if not freqs_cis.is_complex():
                freqs_cis = torch.view_as_complex(
                    freqs_cis.float().reshape(*freqs_cis.shape[:-1], -1, 2)
                )
            seq_len = x.shape[1]
            if freqs_cis.shape[-2] > seq_len:
                freqs_cis = freqs_cis[..., :seq_len, :]
            x_out = torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3)
            return x_out.type_as(x_in)

    def _forward_xpu(self, x, freqs_cis, norm_q_weight, norm_k_weight):
        """XPU path: direct W4A16 GEMM (no activation quantization).

        Handles CPU input transparently: moves to XPU for GEMM, back to input device for output.
        """
        from omni_xpu_kernel import svdq as omni_svdq

        input_device = x.device
        x_orig_dtype = x.dtype
        batch_size, seq_len, channels = x.shape
        xpu_device = cast(torch.Tensor, self.qkv_qweight).device

        x_flat = x.view(batch_size * seq_len, channels)
        if x_flat.device != xpu_device:
            x_flat = x_flat.to(xpu_device)

        # Apply smooth factor (rcp cached)
        smooth = cast(torch.Tensor, self.qkv_smooth_factor)
        if smooth is not None:
            if not hasattr(self, '_xpu_rcp_smooth') or self._xpu_rcp_smooth is None:
                self._xpu_rcp_smooth = (1.0 / smooth.float()).to(torch.float16)
            x_gemm = omni_svdq.fused_smooth_mul_convert(x_flat, self._xpu_rcp_smooth).to(torch.bfloat16)
        else:
            x_gemm = x_flat.to(torch.bfloat16)

        # Prepare oneDNN weights (cached)
        qweight = cast(torch.Tensor, self.qkv_qweight)
        wscales = cast(torch.Tensor, self.qkv_wscales)
        if not hasattr(self, '_xpu_qkv_u4') or self._xpu_qkv_u4 is None:
            self._xpu_qkv_u4, self._xpu_qkv_scales = omni_svdq.prepare_onednn_weights(
                qweight.view(torch.uint8), wscales
            )

        # W4A16 GEMM
        output = omni_svdq.onednn_int4_gemm_preconverted(
            x_gemm, self._xpu_qkv_u4, self._xpu_qkv_scales
        )

        # LoRA
        proj_down = cast(torch.Tensor, self.qkv_proj_down)
        proj_up = cast(torch.Tensor, self.qkv_proj_up)
        if proj_down is not None and proj_up is not None:
            lora_act = x_flat.to(torch.bfloat16) @ proj_down.to(torch.bfloat16)
            lora_out = lora_act @ proj_up.to(torch.bfloat16).T
            output = output.to(torch.bfloat16) + lora_out

        # Bias
        qkv_bias = cast(Optional[torch.Tensor], getattr(self, "qkv_bias", None))
        if qkv_bias is not None:
            output = output.to(torch.bfloat16) + qkv_bias.to(torch.bfloat16)

        # All subsequent ops happen on the same device as the GEMM result
        output = output.to(x_orig_dtype).view(batch_size, seq_len, -1)

        # Split Q, K, V and apply norm + rotary
        query, key, value = output.chunk(3, dim=-1)
        head_dim = int(norm_q_weight.numel())
        heads = query.shape[-1] // head_dim

        query = query.view(batch_size, seq_len, heads, head_dim)
        key = key.view(batch_size, seq_len, heads, head_dim)
        value = value.view(batch_size, seq_len, heads, head_dim)

        query = self._apply_rmsnorm_cpu(query, norm_q_weight)
        key = self._apply_rmsnorm_cpu(key, norm_k_weight)
        if freqs_cis is not None:
            query = self._apply_rotary_emb_cpu(query, freqs_cis)
            key = self._apply_rotary_emb_cpu(key, freqs_cis)

        output = torch.cat(
            [query.flatten(2, 3), key.flatten(2, 3), value.flatten(2, 3)], dim=-1
        )
        return output

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None):
        batch_size, seq_len, channels = x.shape
        norm_q_weight = cast(torch.Tensor, self.norm_q_weight)
        norm_k_weight = cast(torch.Tensor, self.norm_k_weight)

        # XPU: always use direct W4A16 path (better precision)
        if cast(torch.Tensor, self.qkv_qweight).device.type == "xpu":
            return self._forward_xpu(x, freqs_cis, norm_q_weight, norm_k_weight)

        x = x.view(batch_size * seq_len, channels)
        qkv_proj_down = cast(torch.Tensor, self.qkv_proj_down)
        qkv_smooth_factor = cast(torch.Tensor, self.qkv_smooth_factor)
        qkv_qweight = cast(torch.Tensor, self.qkv_qweight)
        qkv_wscales = cast(torch.Tensor, self.qkv_wscales)
        qkv_proj_up = cast(torch.Tensor, self.qkv_proj_up)
        qkv_bias = cast(Optional[torch.Tensor], getattr(self, "qkv_bias", None))
        qkv_wcscales = cast(Optional[torch.Tensor], getattr(self, "qkv_wcscales", None))
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=qkv_proj_down,
            smooth=qkv_smooth_factor,
            fp4=self.qkv_precision == "nvfp4",
            pad_size=256,
        )
        output = torch.empty(
            batch_size * seq_len, self.qkv_out_features, dtype=x.dtype, device=x.device
        )
        if x.device.type == "cpu":
            svdq_gemm_w4a4_cuda(
                act=quantized_x,
                wgt=qkv_qweight,
                out=output,
                ascales=ascales,
                wscales=qkv_wscales,
                lora_act_in=lora_act_out,
                lora_up=qkv_proj_up,
                bias=qkv_bias,
                fp4=self.qkv_precision == "nvfp4",
                alpha=1.0 if self.qkv_precision == "nvfp4" else None,
                wcscales=qkv_wcscales if self.qkv_precision == "nvfp4" else None,
                norm_q=None,
                norm_k=None,
                rotary_emb=None,
            )

            output = output.view(batch_size, seq_len, -1)
            query, key, value = output.chunk(3, dim=-1)
            head_dim = int(norm_q_weight.numel())
            if query.shape[-1] % head_dim != 0:
                raise ValueError(
                    f"query dim {query.shape[-1]} is not divisible by head_dim {head_dim}"
                )
            heads = query.shape[-1] // head_dim

            query = query.view(batch_size, seq_len, heads, head_dim)
            key = key.view(batch_size, seq_len, heads, head_dim)
            value = value.view(batch_size, seq_len, heads, head_dim)

            query = self._apply_rmsnorm_cpu(query, norm_q_weight)
            key = self._apply_rmsnorm_cpu(key, norm_k_weight)
            if freqs_cis is not None:
                query = self._apply_rotary_emb_cpu(query, freqs_cis)
                key = self._apply_rotary_emb_cpu(key, freqs_cis)

            output = torch.cat(
                [query.flatten(2, 3), key.flatten(2, 3), value.flatten(2, 3)], dim=-1
            )
        else:
            svdq_gemm_w4a4_cuda(
                act=quantized_x,
                wgt=qkv_qweight,
                out=output,
                ascales=ascales,
                wscales=qkv_wscales,
                lora_act_in=lora_act_out,
                lora_up=qkv_proj_up,
                bias=qkv_bias,
                fp4=self.qkv_precision == "nvfp4",
                alpha=1.0 if self.qkv_precision == "nvfp4" else None,
                wcscales=qkv_wcscales if self.qkv_precision == "nvfp4" else None,
                norm_q=norm_q_weight,
                norm_k=norm_k_weight,
                rotary_emb=freqs_cis,
            )
            output = output.view(batch_size, seq_len, -1)
        return output


class NunchakuZImageAttention(NunchakuBaseAttention):
    def __init__(self, orig_attn: Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuZImageAttention, self).__init__(processor)
        self.inner_dim = orig_attn.inner_dim
        self.query_dim = orig_attn.query_dim
        self.use_bias = orig_attn.use_bias
        self.dropout = orig_attn.dropout
        self.out_dim = orig_attn.out_dim
        self.context_pre_only = orig_attn.context_pre_only
        self.pre_only = orig_attn.pre_only
        self.heads = orig_attn.heads
        self.rescale_output_factor = orig_attn.rescale_output_factor
        self.is_cross_attention = orig_attn.is_cross_attention

        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k
        with torch.device("meta"):
            to_qkv = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def set_processor(self, processor: str):
        if processor == "flashattn2":
            self.processor = NunchakuZSingleStreamAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


def _convert_z_image_ff(z_ff: ZImageFeedForward) -> FeedForward:
    assert isinstance(z_ff, ZImageFeedForward)
    assert z_ff.w1.in_features == z_ff.w3.in_features
    assert z_ff.w1.out_features == z_ff.w3.out_features
    assert z_ff.w1.out_features == z_ff.w2.in_features
    converted_ff = FeedForward(
        dim=z_ff.w1.in_features,
        dim_out=z_ff.w2.out_features,
        dropout=0.0,
        activation_fn="swiglu",
        inner_dim=z_ff.w2.in_features,
        bias=False,
    ).to(dtype=z_ff.w1.weight.dtype, device=z_ff.w1.weight.device)
    return converted_ff


def replace_fused_module(module, incompatible_keys):
    assert isinstance(module, NunchakuZImageAttention)
    module.fused_module = NunchakuZImageFusedModule(
        module.to_qkv, module.norm_q, module.norm_k
    )
    del module.to_qkv
    del module.norm_q
    del module.norm_k


class NunchakuZImageFeedForward(NunchakuSDXLFeedForward):
    def __init__(self, ff: ZImageFeedForward, **kwargs):
        converted_ff = _convert_z_image_ff(ff)
        NunchakuSDXLFeedForward.__init__(self, converted_ff, **kwargs)


class NunchakuZImageTransformer2DModel(
    ZImageTransformer2DModel, NunchakuModelLoaderMixin
):
    def _patch_model(self, skip_refiners: bool = False, **kwargs):
        def _patch_transformer_block(block_list: List[ZImageTransformerBlock]):
            for _, block in enumerate(block_list):
                block.attention = NunchakuZImageAttention(block.attention, **kwargs)
                block.attention.register_load_state_dict_post_hook(replace_fused_module)
                block.feed_forward = NunchakuZImageFeedForward(
                    block.feed_forward, **kwargs
                )

        def _convert_feed_forward(block_list: List[ZImageTransformerBlock]):
            for _, block in enumerate(block_list):
                block.feed_forward = _convert_z_image_ff(block.feed_forward)

        self.skip_refiners = skip_refiners
        _patch_transformer_block(self.layers)
        if skip_refiners:
            _convert_feed_forward(self.noise_refiner)
            _convert_feed_forward(self.context_refiner)
        else:
            _patch_transformer_block(self.noise_refiner)
            _patch_transformer_block(self.context_refiner)
        return self

    def register_rope_hook(self, rope_hook: NunchakuZImageRopeHook):
        self.rope_hook_handles = []
        for _, ly in enumerate(self.layers):
            self.rope_hook_handles.append(
                ly.attention.register_forward_pre_hook(rope_hook, with_kwargs=True)
            )
        if not self.skip_refiners:
            for _, nr in enumerate(self.noise_refiner):
                self.rope_hook_handles.append(
                    nr.attention.register_forward_pre_hook(rope_hook, with_kwargs=True)
                )
            for _, cr in enumerate(self.context_refiner):
                self.rope_hook_handles.append(
                    cr.attention.register_forward_pre_hook(rope_hook, with_kwargs=True)
                )

    def unregister_rope_hook(self):
        for h in self.rope_hook_handles:
            h.remove()
        self.rope_hook_handles.clear()

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        model_device = next(self.parameters()).device.type
        if model_device in ("cpu", "xpu"):
            # CPU and XPU: use standard diffusers forward (complex freqs_cis)
            # XPU's _forward_xpu handles complex freqs via _apply_rotary_emb_cpu
            return super().forward(
                x, t, cap_feats, patch_size, f_patch_size, return_dict
            )

        # CUDA: register RopeHook to convert freqs to packed format for CUDA kernel
        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            return super().forward(
                x, t, cap_feats, patch_size, f_patch_size, return_dict
            )
        finally:
            self.unregister_rope_hook()
            del rope_hook

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs
    ):
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError(
                "Offload is not supported for ZImageTransformer2DModel"
            )

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert (
            pretrained_model_name_or_path.is_file()
            or pretrained_model_name_or_path.name.endswith((".safetensors", ".sft"))
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(
            pretrained_model_name_or_path, **kwargs
        )
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))

        rank = quantization_config.get("rank", 32)
        skip_refiners = quantization_config.get("skip_refiners", False)
        transformer = transformer.to(torch_dtype)

        precision = get_precision(
            kwargs.get("precision", "auto"), device, pretrained_model_name_or_path
        )
        if precision == "fp4":
            precision = "nvfp4"

        print(
            f"quantization_config: {quantization_config}, rank={rank}, skip_refiners={skip_refiners}"
        )

        transformer._patch_model(
            skip_refiners=skip_refiners, precision=precision, rank=rank, **kwargs
        )
        transformer = transformer.to_empty(device=device)

        device_type = (
            torch.device(device).type
            if not isinstance(device, torch.device)
            else device.type
        )
        if device_type == "cpu" and precision == "int4":
            decode_int4_state_dict_for_cpu(model_state_dict)

        patch_scale_key(transformer, model_state_dict)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, model_state_dict)

        transformer.load_state_dict(model_state_dict)

        return transformer
