# XPU Optimization TODO

Based on unitrace profiling of Z-Image e2e pipeline on Intel Arc B580.

## Completed

### 1. Entire transformer on XPU
- **Impact**: Offload 665ms → Full XPU 149ms (4.5x speedup), 3.23 it/s
- **Root cause fixed**: `NunchakuZImageRopeHook` was converting complex freqs_cis to CUDA-specific packed format. Fix: XPU skips RopeHook, uses CPU complex format.
- **Attention on XPU**: Works correctly after RopeHook fix, no need for CPU fallback.
- **Status**: DONE

### 2. Cache `rcp_smooth` in `_forward_xpu`
- **Impact**: Eliminates per-forward `1.0/smooth_factor` recomputation
- **Status**: DONE (cached as `_xpu_rcp_smooth`)

## Pending Optimizations

### 3. Reduce oneDNN internal M2D overhead
- **Impact**: ~184ms/run (56% of device time) — BIGGEST bottleneck
- **Difficulty**: High (requires oneDNN-level changes)
- **What**: oneDNN INT4 matmul primitive does internal weight reformat (host→device copy) on every call. Even with cached primitive, scratchpad allocation triggers M2D.
- **Possible solutions**:
  - Pre-reformat weights to oneDNN internal layout at model load time
  - Use `onednn_int4_gemm_preconverted` with pre-allocated scratchpad
  - Investigate `dnnl_memory_desc_create_with_tag` for persistent weight format
- **File**: `omni_xpu_kernel/csrc/svdq_onednn.cpp`

### 4. Reduce dtype conversion chain
- **Impact**: ~32ms/run (10% of device time)
- **Difficulty**: Medium
- **What**: Current path: bf16 → f16 (smooth) → bf16 (GEMM input) → f16 (oneDNN output) → f32 (LoRA) → bf16 (output). Could reduce conversions.
- **Trade-off**: f16 GEMM is 3.5x faster than bf16, so bf16→f16 conversion may be worth it.
- **File**: `nunchaku_runtime/models/linear.py` `_forward_xpu()`

### 5. Use `omni_xpu_kernel.sdp` for attention
- **Impact**: Attention is currently only 0.5% of device time (3.5ms), so low priority
- **Difficulty**: High (28min JIT compilation, AOT needed)
- **What**: Replace `dispatch_attention_fn` with `omni_xpu_kernel.sdp.sdp()` (ESIMD Flash Attention).
- **Blockers**: SDP ESIMD requires AOT compilation, 28min JIT on first use, B=1/head_dim=128 constraints.
- **Priority**: LOW (attention is not a bottleneck)

### 6. Use `omni_xpu_kernel.norm` for RMSNorm/LayerNorm
- **Impact**: Small (~1-2% estimated)
- **Difficulty**: Low
- **What**: Replace PyTorch `nn.RMSNorm` with `omni_xpu_kernel.norm.rms_norm()` (ESIMD-optimized).
- **File**: `transformer_zimage.py` `_apply_rmsnorm_cpu()`

### 7. Use `omni_xpu_kernel.rotary` for rotary embeddings
- **Impact**: Small (~1% estimated)
- **Difficulty**: Medium (need to match nunchaku's complex rotary format)
- **What**: Replace complex-number rotary with `omni_xpu_kernel.rotary.rotary_emb()`.
- **File**: `transformer_zimage.py` `_apply_rotary_emb_cpu()`

### 8. Fuse LoRA into oneDNN GEMM via `onednn_int4_gemm_add_to_output`
- **Impact**: ~11ms/run (3% of device time)
- **Difficulty**: Medium
- **What**: Pre-fill dst with LoRA result + bias, then `dst += GEMM(act, wgt)` in one pass.
- **File**: `nunchaku_runtime/models/linear.py` `_forward_xpu()`

### 9. Pipeline-level: overlap text encoder (CPU) with transformer warmup (XPU)
- **Impact**: Reduces total e2e latency (text encoding takes minutes)
- **Difficulty**: Medium
- **What**: Start transformer JIT warmup in parallel with CPU text encoding.

## Profiling Summary (Z-Image 256x256, Arc B580)

### Full XPU mode (current, after optimizations 1-2):
- Transformer forward: **149ms/step** (3.23 it/s for 4 steps)
- Device time breakdown per run:

| Category | Time | % | Notes |
|----------|------|---|-------|
| oneDNN internal M2D | 184ms | 56% | Weight reformat overhead |
| oneDNN GEMM | 57ms | 17% | Core INT4 computation |
| dtype cast | 32ms | 10% | bf16↔f16↔f32 conversions |
| fused_smooth | 21ms | 6% | omni ESIMD kernel |
| Mul (scale) | 13ms | 4% | LoRA/weight scaling |
| Add (residual) | 11ms | 3% | LoRA + bias add |
| XOR (prepare_wgt) | 6ms | 2% | INT4 signed→unsigned |
| Attention SDPA | <1ms | <1% | PyTorch native SDPA |
| **Total** | **~330ms** | | unitrace overhead inflates this |

### Historical comparison:
| Mode | Step time | Speed |
|------|-----------|-------|
| CPU offload (weights XPU, rest CPU) | 665ms | 1.62 it/s |
| Full XPU (attn on CPU) | 350ms | 1.68 it/s |
| **Full XPU (everything on XPU)** | **149ms** | **3.23 it/s** |
