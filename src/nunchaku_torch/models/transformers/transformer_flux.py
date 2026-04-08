import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from diffusers.utils import logging as diffusers_logging
from huggingface_hub import utils

from ...ops.gemm import _C_ops as _GEMM_C_OPS
from ...ops.gemv import _C_ops as _GEMV_C_OPS
from ...ops.quantize import _C_ops as _QUANT_C_OPS
from ...utils import get_precision
from ...ops.fused import fused_gelu_mlp
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.flux import (
    NunchakuFluxAttnProcessor,
    NunchakuFluxSingleAttnProcessor,
)
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import (
    NunchakuModelLoaderMixin,
    convert_fp16,
    decode_int4_state_dict_for_cpu,
    patch_scale_key,
)

logger = diffusers_logging.get_logger(__name__)


def _copy_attn_attrs(dst, src):
    """Copy common attention attributes from src to dst."""
    for attr in (
        "inner_dim", "query_dim", "use_bias", "dropout", "fused_projections",
        "out_dim", "context_pre_only", "pre_only", "heads", "head_dim",
        "added_kv_proj_dim", "added_proj_bias",
    ):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


class NunchakuFluxAttention(NunchakuBaseAttention):
    """Quantized Flux joint attention with fused QKV."""

    def __init__(self, other, processor: str = "nunchaku", **kwargs):
        super().__init__(processor)
        _copy_attn_attrs(self, other)
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.norm_added_q = getattr(other, "norm_added_q", None)
        self.norm_added_k = getattr(other, "norm_added_k", None)

        # Fuse Q/K/V into single quantized linear
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

        self.to_out = other.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)

        # Context projections (joint attention)
        if self.added_kv_proj_dim is not None:
            with torch.device("meta"):
                add_qkv_proj = fuse_linears(
                    [other.add_q_proj, other.add_k_proj, other.add_v_proj]
                )
            self.add_qkv_proj = SVDQW4A4Linear.from_linear(add_qkv_proj, **kwargs)
            self.to_add_out = SVDQW4A4Linear.from_linear(other.to_add_out, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )

    def set_processor(self, processor: str):
        if processor in ("nunchaku", "flashattn2"):
            self.processor = NunchakuFluxAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuFluxSingleAttention(NunchakuBaseAttention):
    """Quantized Flux single-stream attention with fused QKV."""

    def __init__(self, other, processor: str = "nunchaku", **kwargs):
        super().__init__(processor)
        _copy_attn_attrs(self, other)
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k

        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )

    def set_processor(self, processor: str):
        if processor in ("nunchaku", "flashattn2"):
            self.processor = NunchakuFluxSingleAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuFluxTransformerBlock(FluxTransformerBlock):
    """Quantized Flux joint transformer block."""

    def __init__(self, other: FluxTransformerBlock, **kwargs):
        super(FluxTransformerBlock, self).__init__()

        # AdaLayerNormZero - quantize the linear projection
        self.norm1 = other.norm1
        self.norm1.linear = AWQW4A16Linear.from_linear(other.norm1.linear, **kwargs)
        self.norm1_context = other.norm1_context
        self.norm1_context.linear = AWQW4A16Linear.from_linear(
            other.norm1_context.linear, **kwargs
        )

        # Attention
        self.attn = NunchakuFluxAttention(other.attn, **kwargs)

        # FF norm + MLP
        self.norm2 = other.norm2
        self.ff = NunchakuFeedForward(other.ff, **kwargs)
        self.norm2_context = other.norm2_context
        self.ff_context = NunchakuFeedForward(other.ff_context, **kwargs)

    @staticmethod
    def _nunchaku_ada_norm(norm_module, x, temb):
        """AdaLayerNormZero using nunchaku C++ convention: x * scale + shift.

        The C++ ``split_mod<6>`` kernel splits the embedding output with an
        interleaved (strided) pattern — element *i* of component *k* comes
        from position ``i*6 + k``.  ``torch.chunk`` uses contiguous slices,
        so we reshape to ``(..., 6)`` and unbind to match the C++ behavior.
        """
        emb = norm_module.linear(norm_module.silu(temb))
        # Interleaved split matching C++ split_mod<6>
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            emb.unflatten(-1, (-1, 6)).unbind(-1)
        )
        norm_x = norm_module.norm(x) * scale_msa[:, None] + shift_msa[:, None]
        return norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _nunchaku_ada_norm_single(norm_module, x, temb):
        """AdaLayerNormZeroSingle using nunchaku C++ convention."""
        emb = norm_module.linear(norm_module.silu(temb))
        # Interleaved split matching C++ split_mod<3>
        shift_msa, scale_msa, gate_msa = emb.unflatten(-1, (-1, 3)).unbind(-1)
        norm_x = norm_module.norm(x) * scale_msa[:, None] + shift_msa[:, None]
        return norm_x, gate_msa

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._nunchaku_ada_norm(self.norm1, hidden_states, temb)
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self._nunchaku_ada_norm(self.norm1_context, encoder_hidden_states, temb)
        )

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Image stream
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * scale_mlp[:, None] + shift_mlp[:, None]
        )
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Context stream
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * c_scale_mlp[:, None]
            + c_shift_mlp[:, None]
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NunchakuFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    """Quantized Flux single-stream transformer block.

    Uses decomposed layout matching the nunchaku checkpoint:
      out_proj(attn) + mlp_fc2(gelu(mlp_fc1(x)))
    instead of diffusers' proj_out(cat[attn, gelu(proj_mlp(x))]).
    These are mathematically equivalent.
    """

    def __init__(self, other: FluxSingleTransformerBlock, **kwargs):
        super(FluxSingleTransformerBlock, self).__init__()

        # AdaLayerNormZeroSingle
        self.norm = other.norm
        self.norm.linear = AWQW4A16Linear.from_linear(other.norm.linear, **kwargs)

        # Attention (single-stream, no cross-attention)
        self.attn = NunchakuFluxSingleAttention(other.attn, **kwargs)

        # Decomposed MLP + output projection (matching checkpoint layout)
        # mlp_fc1: 3072 → 12288, mlp_fc2: 12288 → 3072
        self.mlp_fc1 = SVDQW4A4Linear.from_linear(other.proj_mlp, **kwargs)
        self.act_mlp = other.act_mlp
        inner_dim = other.proj_mlp.out_features  # 12288
        out_dim = other.proj_out.out_features  # 3072
        # mlp_fc2: takes fused GELU output (12288) → 3072
        with torch.device("meta"):
            mlp_fc2_placeholder = nn.Linear(inner_dim, out_dim)
        self.mlp_fc2 = SVDQW4A4Linear.from_linear(mlp_fc2_placeholder, **kwargs)
        # fc2 receives unsigned-quantized GELU output (with shift)
        self.mlp_fc2.act_unsigned = self.mlp_fc2.precision != "nvfp4"
        # out_proj: takes attn output (3072) → 3072
        attn_dim = other.proj_out.out_features  # 3072
        with torch.device("meta"):
            out_proj_placeholder = nn.Linear(attn_dim, out_dim)
        self.out_proj = SVDQW4A4Linear.from_linear(out_proj_placeholder, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = NunchakuFluxTransformerBlock._nunchaku_ada_norm_single(
            self.norm, hidden_states, temb
        )

        # MLP path: fused fc1 → GELU → fc2 (same fusion as NunchakuFeedForward)
        mlp_output = fused_gelu_mlp(norm_hidden_states, self.mlp_fc1, self.mlp_fc2)

        # Attention path
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        attn_proj = self.out_proj(attn_output)

        # Combine: equivalent to proj_out(cat[attn, mlp])
        gate = gate.unsqueeze(1)
        hidden_states = residual + gate * (attn_proj + mlp_output)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


def _remap_flux_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap nunchaku checkpoint keys to match our model structure."""

    # Parameter suffix remapping (applied to all SVDQ layers)
    _PARAM_REMAP = {
        ".smooth": ".smooth_factor",
        ".smooth_orig": ".smooth_factor_orig",
        ".lora_up": ".proj_up",
        ".lora_down": ".proj_down",
    }

    # Structural key remapping for joint blocks
    _JOINT_REMAP = {
        "qkv_proj.": "attn.to_qkv.",
        "qkv_proj_context.": "attn.add_qkv_proj.",
        "out_proj_context.": "attn.to_add_out.",
        "out_proj.": "attn.to_out.0.",
        "norm_q.": "attn.norm_q.",
        "norm_k.": "attn.norm_k.",
        "norm_added_q.": "attn.norm_added_q.",
        "norm_added_k.": "attn.norm_added_k.",
        "mlp_fc1.": "ff.net.0.proj.",
        "mlp_fc2.": "ff.net.2.",
        "mlp_context_fc1.": "ff_context.net.0.proj.",
        "mlp_context_fc2.": "ff_context.net.2.",
    }

    # Structural key remapping for single blocks
    _SINGLE_REMAP = {
        "qkv_proj.": "attn.to_qkv.",
        "norm_q.": "attn.norm_q.",
        "norm_k.": "attn.norm_k.",
        # mlp_fc1, mlp_fc2, out_proj stay as-is (match our restructured block)
    }

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # Apply structural remapping
        if key.startswith("transformer_blocks."):
            parts = key.split(".", 2)  # ['transformer_blocks', 'N', 'rest']
            if len(parts) == 3:
                prefix = f"{parts[0]}.{parts[1]}."
                rest = parts[2]
                for old, new in _JOINT_REMAP.items():
                    if rest.startswith(old):
                        new_key = prefix + new + rest[len(old):]
                        break

        elif key.startswith("single_transformer_blocks."):
            parts = key.split(".", 2)
            if len(parts) == 3:
                prefix = f"{parts[0]}.{parts[1]}."
                rest = parts[2]
                for old, new in _SINGLE_REMAP.items():
                    if rest.startswith(old):
                        new_key = prefix + new + rest[len(old):]
                        break

        # Apply parameter suffix remapping
        for old_suffix, new_suffix in _PARAM_REMAP.items():
            if new_key.endswith(old_suffix):
                new_key = new_key[: -len(old_suffix)] + new_suffix
                break

        new_state_dict[new_key] = value

    return new_state_dict


class NunchakuFluxTransformer2DModel(FluxTransformer2DModel, NunchakuModelLoaderMixin):
    """Quantized Flux transformer with CPU support."""

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, ...] = (16, 56, 56),
    ):
        self._is_initialized = False
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

    def _patch_model(self, **kwargs):
        for i, block in enumerate(self.transformer_blocks):
            self.transformer_blocks[i] = NunchakuFluxTransformerBlock(block, **kwargs)
        for i, block in enumerate(self.single_transformer_blocks):
            self.single_transformer_blocks[i] = NunchakuFluxSingleTransformerBlock(
                block, **kwargs
            )
        # Keep original FluxPosEmbed — it returns (cos, sin) tuple expected by
        # diffusers' apply_rotary_emb.
        self._is_initialized = True
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs
    ):
        device = kwargs.get("device", "cpu")
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        cpu_kernel_layout = kwargs.pop("cpu_kernel_layout", True)

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
        transformer = transformer.to(torch_dtype)

        precision = get_precision(
            kwargs.get("precision", "auto"), device, pretrained_model_name_or_path
        )
        if precision == "fp4":
            precision = "nvfp4"

        transformer._patch_model(precision=precision, rank=rank, torch_dtype=torch_dtype)
        transformer = transformer.to_empty(device=device)

        # Remap checkpoint keys to match our model structure
        model_state_dict = _remap_flux_state_dict(model_state_dict)

        device_type = (
            torch.device(device).type
            if not isinstance(device, torch.device)
            else device.type
        )
        needs_cpu_int4_layout = (
            device_type in ("cpu", "xpu")
            and precision == "int4"
            and (
                cpu_kernel_layout
                or _GEMM_C_OPS is None
                or _QUANT_C_OPS is None
                or _GEMV_C_OPS is None
            )
        )
        if needs_cpu_int4_layout:
            decode_int4_state_dict_for_cpu(model_state_dict)

        patch_scale_key(transformer, model_state_dict)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, model_state_dict)

        result = transformer.load_state_dict(model_state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"Missing keys: {result.missing_keys[:10]}...")
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys: {result.unexpected_keys[:10]}...")
        return transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        # Use no_grad to prevent autograd from accumulating intermediate
        # tensors across 57 blocks — without this, CPU inference OOMs.
        with torch.no_grad():
            return self._forward_impl(
                hidden_states, encoder_hidden_states, pooled_projections,
                timestep, img_ids, txt_ids, guidance, joint_attention_kwargs,
                controlnet_block_samples, controlnet_single_block_samples,
                return_dict,
            )

    def _forward_impl(
        self,
        hidden_states, encoder_hidden_states, pooled_projections,
        timestep, img_ids, txt_ids, guidance, joint_attention_kwargs,
        controlnet_block_samples, controlnet_single_block_samples,
        return_dict,
    ):
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        if guidance is not None and hasattr(self.time_text_embed, "guidance_embedder"):
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(
                    controlnet_block_samples
                )
                interval_control = int(max(interval_control, 1))
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[index_block // interval_control]
                )

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(max(interval_control, 1))
                hidden_states = (
                    hidden_states
                    + controlnet_single_block_samples[
                        index_block // interval_control
                    ]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def to(self, *args, **kwargs):
        device_arg_or_kwarg_present = (
            any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        )
        dtype_present_in_args = "dtype" in kwargs
        for arg in args:
            if isinstance(arg, str):
                try:
                    torch.device(arg)
                    device_arg_or_kwarg_present = True
                except RuntimeError:
                    pass
            if isinstance(arg, torch.dtype):
                dtype_present_in_args = True

        if dtype_present_in_args and self._is_initialized:
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. "
                "Please use the `torch_dtype` argument in `from_pretrained`."
            )
        return super(type(self), self).to(*args, **kwargs)
