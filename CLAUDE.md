# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Claude Code skill project** (`cuda-kernels`) for writing and benchmarking optimized CUDA kernels on NVIDIA GPUs (H100, A100, T4). It provides integration patterns for HuggingFace **diffusers** (video/image generation) and **transformers** (LLMs), plus support for loading pre-compiled kernels from HuggingFace Hub via the `kernels` library.

This is not a traditional Python package — it's a skill installed at `.claude/skills/cuda-kernels/` containing documentation, reference guides, and example scripts.

## Running Examples

```bash
# Diffusers integration (LTX-Video kernel injection)
python .claude/skills/cuda-kernels/scripts/ltx_kernel_injection_example.py

# Transformers integration (LLaMA/Mistral RMSNorm)
python .claude/skills/cuda-kernels/scripts/transformers_injection_example.py

# HuggingFace Kernels Hub usage
python .claude/skills/cuda-kernels/scripts/huggingface_kernels_example.py

# Micro-benchmark (RMSNorm custom vs PyTorch)
python .claude/skills/cuda-kernels/scripts/benchmark_rmsnorm.py

# End-to-end video generation benchmark
python .claude/skills/cuda-kernels/scripts/benchmark_example.py --use-optimized-kernels
```

## Architecture

### Skill Structure

- **SKILL.md** — Skill metadata, quick start, GPU reference tables
- **scripts/** — 5 working Python examples covering diffusers, transformers, Hub integration, and benchmarking
- **references/** — Deep technical guides: GPU-specific optimization (H100/A100/T4), kernel templates, integration patterns, troubleshooting

### Key Integration Patterns

**Module detection** — Use `type(module).__name__ == 'RMSNorm'` (not `isinstance()`) because diffusers' RMSNorm doesn't inherit from `torch.nn.RMSNorm`.

**Kernel injection timing** — Must happen AFTER `pipe.to("cuda")` but BEFORE `enable_model_cpu_offload()`.

**Weight handling** — Always check `hasattr(module, 'weight') and module.weight is not None`. Diffusers RMSNorm may have `elementwise_affine=False` (no weight); all transformers RMSNorm modules have weights.

**Epsilon detection** — LLaMA uses `variance_epsilon`; others use `eps`. Fall back to `1e-6`.

**Forward patching** — Replace `module.forward` with a closure via factory function (`make_patched_forward()`) rather than modifying model structure.

### CUDA Kernel Conventions

- Every `.cu` file must include explicit type conversion helpers (`to_float`/`from_float`) because PyTorch compiles with `-D__CUDA_NO_HALF_OPERATORS__` which disables implicit FP16/BF16 conversions.
- Use vectorized memory access (`__nv_bfloat162`, `__half2`, `float4`) for bandwidth-bound kernels.
- T4 has **no BFloat16 support** — must use FP16. T4 also has half the shared memory (64 KB) and max threads/SM (1024) vs H100/A100.
- Custom CUDA kernels and `torch.compile` are mutually exclusive unless registered as a PyTorch custom op via `@torch.library.custom_op()`.

## Installing the Skill

```bash
kernels skills add --claude          # Project-level
kernels skills add --claude --global # User-level
```
