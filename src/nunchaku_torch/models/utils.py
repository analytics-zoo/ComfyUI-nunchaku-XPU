import copy

import torch
from torch import nn

from ..utils import copy_params_into


def _device_event(device, **kwargs):
    """Create a device Event (CUDA or XPU)."""
    if device.type == "xpu":
        # torch.xpu.Event doesn't support 'blocking' kwarg
        kwargs.pop("blocking", None)
        return torch.xpu.Event(**kwargs)
    return torch.cuda.Event(**kwargs)


def _device_stream(device):
    """Create a device Stream (CUDA or XPU)."""
    if device.type == "xpu":
        return torch.xpu.Stream(device=device)
    return torch.cuda.Stream(device=device)


def _current_stream(device):
    """Get current stream for device."""
    if device.type == "xpu":
        return torch.xpu.current_stream(device)
    return torch.cuda.current_stream(device)


def _stream_context(stream):
    """Context manager for stream (works for both CUDA and XPU)."""
    if hasattr(torch.xpu, "stream") and isinstance(stream, torch.xpu.Stream):
        return torch.xpu.stream(stream)
    return torch.cuda.stream(stream)


def _empty_cache(device):
    """Empty cache for device."""
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def fuse_linears(linears: list[nn.Linear]) -> nn.Linear:
    assert len(linears) > 0
    if len(linears) == 1:
        return linears[0]
    else:
        assert all(linear.in_features == linears[0].in_features for linear in linears)
        out_features = sum(linear.out_features for linear in linears)
        bias = all(linear.bias is not None for linear in linears)
        return nn.Linear(
            linears[0].in_features,
            out_features,
            bias=bias,
            dtype=linears[0].weight.dtype,
            device=linears[0].weight.device,
        )


class CPUOffloadManager:
    def __init__(
        self,
        blocks: list[nn.Module],
        device: str | torch.device = torch.device("cuda"),
        use_pin_memory: bool = True,
        on_gpu_modules: list[nn.Module] = [],
        num_blocks_on_gpu: int = 1,
        empty_cache_freq: int = 0,
    ):
        self.blocks = blocks
        self.use_pin_memory = use_pin_memory
        self.on_gpu_modules = on_gpu_modules
        self.num_blocks_on_gpu = num_blocks_on_gpu
        assert self.num_blocks_on_gpu > 0

        self.memory_stream = None
        self._device_obj = None  # set in set_device

        # Events created lazily in set_device once we know the device type
        self.compute_done = None
        self.memory_done = None

        self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]

        self.device = None
        self.set_device(device)

        self.current_block_idx = 0
        self.forward_counter = 0
        self.empty_cache_freq = empty_cache_freq

    def set_device(self, device: torch.device | str, force: bool = False):
        if isinstance(device, str):
            device = torch.device(device)
        assert device.type in ("cuda", "xpu")
        if self.device == device and not force:
            return
        self.device = device
        self._device_obj = device
        self.memory_stream = _device_stream(device)
        self.compute_done = _device_event(device, blocking=False)
        self.memory_done = _device_event(device, blocking=False)
        for block in self.buffer_blocks:
            block.to(device)
        for module in self.on_gpu_modules:
            module.to(device)
        for i, block in enumerate(self.blocks):
            if i < self.num_blocks_on_gpu:
                block.to(device)
            else:
                block.to("cpu")
                if self.use_pin_memory:
                    for p in block.parameters(recurse=True):
                        p.data = p.data.pin_memory()
                    for b in block.buffers(recurse=True):
                        b.data = b.data.pin_memory()

    def load_block(self, block_idx: int, non_blocking: bool = True):
        if block_idx < self.num_blocks_on_gpu:
            return
        if block_idx >= len(self.blocks):
            return

        block = self.blocks[block_idx]
        copy_params_into(
            block, self.buffer_blocks[block_idx % 2], non_blocking=non_blocking
        )

    def step(self, compute_stream=None):
        if compute_stream is None:
            compute_stream = _current_stream(self.device)
        next_compute_done = _device_event(self.device)
        next_compute_done.record(compute_stream)
        with _stream_context(self.memory_stream):
            self.memory_stream.wait_event(self.compute_done)
            self.load_block(self.current_block_idx + 1)
            next_memory_done = _device_event(self.device)
            next_memory_done.record(self.memory_stream)
        self.memory_done = next_memory_done
        self.compute_done = next_compute_done
        self.current_block_idx += 1
        if self.current_block_idx < len(self.blocks):
            compute_stream.wait_event(self.memory_done)
        else:
            compute_stream.wait_event(self.compute_done)
            self.current_block_idx = 0
            self.forward_counter += 1
            if (
                self.empty_cache_freq > 0
                and self.forward_counter % self.empty_cache_freq == 0
            ):
                _empty_cache(self.device)

    def get_block(self, block_idx: int | None = None) -> nn.Module:
        if block_idx is None:
            block_idx = self.current_block_idx
        if block_idx < self.num_blocks_on_gpu:
            return self.blocks[block_idx]
        else:
            return self.buffer_blocks[block_idx % 2]

    def initialize(self, stream=None):
        if stream is None:
            stream = _current_stream(self.device)
        self.compute_done.record(stream)
        self.memory_done.record(stream)
