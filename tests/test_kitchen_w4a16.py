"""Integration tests for the Comfy Kitchen managed W4A16 linear route."""

from __future__ import annotations

import pytest
import torch


def _kitchen_w4a16_available() -> bool:
    try:
        import comfy_kitchen as ck
        from comfy_kitchen.backends import xpu

        return bool(
            torch.xpu.is_available()
            and ck.list_backends()["xpu"]["available"]
            and xpu._SVDQ_W4A16_AVAILABLE
        )
    except (AttributeError, ImportError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _kitchen_w4a16_available(),
    reason="Comfy Kitchen SVDQuant W4A16 XPU backend is unavailable",
)


def _make_linear(*, m=31, n=64, k=128, rank=8):
    from nunchaku_torch.models.linear import SVDQW4A4Linear

    device = torch.device("xpu:0")
    generator = torch.Generator(device=device).manual_seed(20260728)
    linear = SVDQW4A4Linear(
        k,
        n,
        rank=rank,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    linear.qweight.data.copy_(
        torch.randint(
            -128,
            128,
            linear.qweight.shape,
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
    )
    linear.wscales.data.copy_(
        (
            torch.rand(
                linear.wscales.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.05
            + 0.005
        ).to(torch.bfloat16)
    )
    linear.smooth_factor.data.copy_(
        (
            torch.rand(
                linear.smooth_factor.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.5
            + 0.75
        ).to(torch.bfloat16)
    )
    linear.proj_down.data.copy_(
        (
            torch.randn(
                linear.proj_down.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.02
        ).to(torch.bfloat16)
    )
    linear.proj_up.data.copy_(
        (
            torch.randn(
                linear.proj_up.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.02
        ).to(torch.bfloat16)
    )
    linear.bias.data.copy_(
        (
            torch.randn(
                linear.bias.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.01
        ).to(torch.bfloat16)
    )
    x = (
        torch.randn(
            1,
            m,
            k,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16)
    return linear, x


def _direct_previous_route(linear, x, signed_qweight):
    from omni_xpu_kernel import svdq

    x_flat = x.reshape(-1, x.shape[-1])
    rcp_smooth = (1.0 / linear.smooth_factor.float()).to(torch.float16)
    x_gemm = svdq.fused_smooth_mul_convert(x_flat, rcp_smooth)
    x_gemm.nan_to_num_(
        nan=0.0,
        posinf=65504.0,
        neginf=-65504.0,
    )
    packed_u4, scales_f16 = svdq.prepare_onednn_weights(
        signed_qweight.view(torch.uint8),
        linear.wscales,
    )
    dst = torch.zeros(
        x_flat.shape[0],
        linear.out_features,
        dtype=torch.bfloat16,
        device=x.device,
    )
    lora = (
        x_flat.to(torch.bfloat16)
        @ linear.proj_down.to(torch.bfloat16)
    ) @ linear.proj_up.to(torch.bfloat16).t()
    dst.add_(lora)
    dst.add_(linear.bias.to(torch.bfloat16))
    svdq.onednn_int4_gemm_add_to_output(
        x_gemm,
        packed_u4,
        scales_f16,
        dst,
    )
    return dst.to(x.dtype).reshape(x.shape[0], x.shape[1], -1)


def test_linear_uses_kitchen_and_matches_previous_route_byte_for_byte():
    import comfy_kitchen as ck

    linear, x = _make_linear()
    signed_qweight = linear.qweight.detach().clone()
    expected = _direct_previous_route(linear, x, signed_qweight)

    ck.get_svdquant_w4a16_route_diagnostics(reset=True)
    with torch.no_grad():
        actual = linear(x)
    torch.xpu.synchronize()

    prepared = linear._xpu_w4a16_prepared
    assert prepared is not None
    assert prepared.packed_u4.data_ptr() == linear.qweight.data_ptr()
    assert torch.equal(
        linear.qweight.view(torch.uint8),
        signed_qweight.view(torch.uint8) ^ 0x88,
    )
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert ck.get_svdquant_w4a16_route_diagnostics(reset=True) == {
        "routes": {"xpu": 1},
        "fallbacks": {},
    }


def test_state_dict_restores_checkpoint_bytes_and_reprepares():
    linear, x = _make_linear()
    signed_qweight = linear.qweight.detach().clone()
    with torch.no_grad():
        first = linear(x)
    assert linear._xpu_w4a16_prepared is not None

    state = linear.state_dict()
    assert linear._xpu_w4a16_prepared is None
    assert torch.equal(linear.qweight, signed_qweight)
    assert torch.equal(state["qweight"], signed_qweight)

    with torch.no_grad():
        second = linear(x)
    assert linear._xpu_w4a16_prepared is not None
    assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))


def test_load_state_dict_invalidates_prepared_storage():
    linear, x = _make_linear()
    state = {name: value.clone() for name, value in linear.state_dict().items()}
    replacement = torch.bitwise_xor(state["qweight"].view(torch.uint8), 0x11)
    state["qweight"] = replacement.view(torch.int8)

    with torch.no_grad():
        linear(x)
    assert linear._xpu_w4a16_prepared is not None

    linear.load_state_dict(state)
    assert linear._xpu_w4a16_prepared is None
    assert torch.equal(linear.qweight, state["qweight"])


def test_module_move_restores_signed_source_before_apply():
    linear, x = _make_linear()
    signed_qweight = linear.qweight.detach().clone().cpu()
    with torch.no_grad():
        linear(x)
    assert linear._xpu_w4a16_prepared is not None

    linear.cpu()
    assert linear._xpu_w4a16_prepared is None
    assert torch.equal(linear.qweight, signed_qweight)


def test_runtime_switch_to_w4a4_restores_source(monkeypatch):
    from nunchaku_torch.models.linear import SVDQW4A4Linear

    linear, x = _make_linear()
    signed_qweight = linear.qweight.detach().clone()
    with torch.no_grad():
        linear(x)
    assert linear._xpu_w4a16_prepared is not None

    def fake_w4a4(self, value):
        assert self._xpu_w4a16_prepared is None
        assert torch.equal(self.qweight, signed_qweight)
        return torch.zeros(
            value.shape[0],
            self.out_features,
            dtype=value.dtype,
            device=value.device,
        )

    monkeypatch.setattr(SVDQW4A4Linear, "_forward_xpu_w4a4", fake_w4a4)
    linear._xpu_force_w4a4 = True
    with torch.no_grad():
        output = linear(x)
    assert output.shape == (x.shape[0], x.shape[1], linear.out_features)
