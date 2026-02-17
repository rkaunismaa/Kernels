# CUDA Kernels Demo

Hands-on exploration of optimized CUDA kernels using the [HuggingFace `kernels`](https://huggingface.co/docs/kernels) package. Demonstrates downloading pre-compiled GPU kernels from HuggingFace Hub, benchmarking them against PyTorch, and patching real model architectures — all on a local RTX 4090.

Also includes a [Claude Code skill](https://docs.anthropic.com/en/docs/claude-code/skills) (`cuda-kernels`) with reference guides for writing and integrating custom CUDA kernels targeting HuggingFace diffusers and transformers.

## Notebooks

### [demo_4090.ipynb](demo_4090.ipynb) — Beginner

A first introduction to CUDA kernels for someone with zero prior knowledge.

- What a CUDA kernel is and why it matters
- Downloading a pre-compiled GELU activation kernel from HuggingFace Hub
- Benchmarking it against PyTorch's built-in GELU
- Verifying correctness (both produce the same numbers)
- Memory bandwidth analysis — why GELU is memory-bound

### [demo_4090_advanced.ipynb](demo_4090_advanced.ipynb) — Advanced

Deep dive into kernel fusion, the technique that actually delivers real speedups.

- **Kernel fusion** — why combining `SiLU + multiply` into one GPU call gives ~1.4x speedup (eliminates intermediate memory round-trips)
- **RMSNorm** — standalone replacement vs. fused residual+norm, showing where each wins and loses
- **Fused LayerNorm + Linear** — merging normalization and matrix multiplication so the intermediate tensor never touches memory
- **Model patching** — building a LLaMA-style MLP block and surgically replacing its internals with Hub kernels (the same technique used by vLLM and TGI)
- **Scaling analysis** — how speedups change with token count, revealing the launch overhead vs. memory bandwidth tradeoff
- **Memory bandwidth deep dive** — quantifying bytes saved per fusion and achieved vs. theoretical bandwidth

## Quick Start

```bash
# Create a virtual environment with uv
uv venv .kernels --python 3.12
uv pip install --python .kernels/bin/python torch kernels matplotlib scipy

# Run the standalone demo script
CUDA_VISIBLE_DEVICES=0 .kernels/bin/python demo_4090.py

# Or open the notebooks
jupyter notebook demo_4090.ipynb
```

## Requirements

- NVIDIA GPU with CUDA support (tested on RTX 4090)
- Python 3.12+
- PyTorch 2.x with CUDA
- `kernels` package (HuggingFace)
- `matplotlib` and `scipy` (for notebook visualizations)

## Claude Code Skill

The `.claude/skills/cuda-kernels/` directory contains a Claude Code skill with:

| Directory | Contents |
|-----------|----------|
| `scripts/` | 5 working examples — diffusers integration, transformers integration, Hub kernel usage, micro-benchmarks, end-to-end video generation benchmark |
| `references/` | GPU optimization guides ([H100](.claude/skills/cuda-kernels/references/h100-optimization-guide.md), [A100](.claude/skills/cuda-kernels/references/a100-optimization-guide.md), [T4](.claude/skills/cuda-kernels/references/t4-optimization-guide.md)), [kernel templates](.claude/skills/cuda-kernels/references/kernel-templates.md), integration guides ([diffusers](.claude/skills/cuda-kernels/references/diffusers-integration.md), [transformers](.claude/skills/cuda-kernels/references/transformers-integration.md)), [troubleshooting](.claude/skills/cuda-kernels/references/troubleshooting.md) |

Install the skill into your own project:

```bash
pip install kernels
kernels skills add --claude
```

## Key Findings (RTX 4090)

- **Fused `silu_and_mul`**: ~1.4x faster than separate `SiLU` + multiply
- **Fused RMSNorm + residual**: ~2x faster at 8K+ tokens
- **Standalone RMSNorm replacement**: slower — PyTorch's native kernel is already well-optimized on the 4090
- **End-to-end MLP block**: ~2% faster after patching (linear projections dominate 95% of runtime)
- **Takeaway**: fusion matters more than drop-in replacement; speedups are size-dependent; these kernels show larger gains on datacenter GPUs (H100/A100)
