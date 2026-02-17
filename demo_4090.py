#!/usr/bin/env python3
"""
Simple demo: Race an optimized CUDA kernel vs PyTorch on a 4090.

What this does:
  1. Downloads a pre-compiled GELU activation kernel from HuggingFace Hub
  2. Creates a big tensor (matrix of numbers) on the GPU
  3. Runs both the Hub kernel and PyTorch's built-in GELU on it
  4. Compares speed and verifies they produce the same result
"""

import time
import torch
from kernels import get_kernel, has_kernel


def main():
    # --- Step 1: Show what GPU we're working with ---
    print("=" * 60)
    print("CUDA Kernel Demo on RTX 4090")
    print("=" * 60)
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print()

    # --- Step 2: Download the optimized kernel from HuggingFace Hub ---
    #
    # This is like pip install, but for GPU code. The "kernels" library
    # downloads a pre-compiled CUDA kernel that was built and optimized
    # for your specific GPU architecture.
    #
    repo_id = "kernels-community/activation"
    print(f"Downloading optimized kernel: {repo_id}")

    if not has_kernel(repo_id):
        print("No compatible kernel found for this GPU. Exiting.")
        return

    kernel = get_kernel(repo_id)
    print(f"Kernel loaded! Available functions: "
          f"{[f for f in dir(kernel) if not f.startswith('_')]}")
    print()

    # --- Step 3: Create test data on the GPU ---
    #
    # We create a large matrix of random numbers directly on the GPU.
    # float16 is the precision most AI models use during inference.
    #
    sizes = [
        (1024, 2048),    # ~2 million numbers   (small)
        (4096, 4096),    # ~16 million numbers   (medium)
        (8192, 8192),    # ~67 million numbers   (large)
    ]

    print(f"{'Size':>16}  {'Hub kernel':>12}  {'PyTorch':>12}  {'Speedup':>8}  {'Match?':>6}")
    print("-" * 62)

    for size in sizes:
        x = torch.randn(size, dtype=torch.float16, device="cuda")
        y_hub = torch.empty_like(x)

        # --- Step 4: Warm up both implementations ---
        #
        # GPUs need a few "warm-up" runs to reach peak performance.
        # The first run is always slower because the GPU has to set up
        # memory, load the kernel code, etc.
        #
        for _ in range(10):
            kernel.gelu_fast(y_hub, x)
            torch.nn.functional.gelu(x)
        torch.cuda.synchronize()  # Wait for GPU to finish

        # --- Step 5: Benchmark the Hub kernel ---
        #
        # We run each 200 times and measure the average.
        # torch.cuda.synchronize() forces us to wait for the GPU,
        # otherwise Python would race ahead while the GPU is still working.
        #
        iterations = 200
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            kernel.gelu_fast(y_hub, x)
        torch.cuda.synchronize()
        hub_ms = (time.perf_counter() - start) / iterations * 1000

        # --- Step 6: Benchmark PyTorch's built-in GELU ---
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            y_torch = torch.nn.functional.gelu(x)
        torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - start) / iterations * 1000

        # --- Step 7: Verify correctness ---
        #
        # The whole point is that both should produce the SAME result.
        # A fast kernel that gives wrong answers is useless.
        # We rerun once to get fresh outputs for comparison.
        #
        kernel.gelu_fast(y_hub, x)
        y_torch = torch.nn.functional.gelu(x)
        torch.cuda.synchronize()
        max_diff = (y_hub - y_torch).abs().max().item()
        match = "YES" if max_diff < 0.01 else "NO"

        speedup = torch_ms / hub_ms
        num_elements = size[0] * size[1]
        print(f"{num_elements:>13,}    {hub_ms:>9.4f} ms  {torch_ms:>9.4f} ms  {speedup:>6.2f}x  {match:>6}")

    print()
    print("What just happened:")
    print("  - We downloaded a pre-compiled GELU kernel from HuggingFace")
    print("  - Ran it on your 4090 against PyTorch's built-in version")
    print("  - Both produce the same numbers (correctness check)")
    print("  - The speedup shows whether the hand-optimized kernel beats PyTorch")
    print()
    print("In a real AI model, GELU runs thousands of times per inference.")
    print("Even small per-call speedups compound into meaningful time savings.")


if __name__ == "__main__":
    main()
