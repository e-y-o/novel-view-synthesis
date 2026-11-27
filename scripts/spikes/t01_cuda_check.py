"""T01: verify PyTorch runs real work on the GPU (Blackwell sm_120).

Usage: uv run python scripts/spikes/t01_cuda_check.py [--out PATH]
"""
import argparse
import platform
import statistics
import sys
import time

import torch


def timed(fn, warmup=2, runs=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="also write the report to this file")
    args = ap.parse_args()
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    if not torch.cuda.is_available():
        sys.exit("CUDA not available: check the NVIDIA driver (nvidia-smi) first.")
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    arch = torch.cuda.get_arch_list()
    say(f"Python:        {platform.python_version()}")
    say(f"torch:         {torch.__version__}")
    say(f"torch CUDA:    {torch.version.cuda}")
    say(f"device:        {torch.cuda.get_device_name(0)}")
    say(f"capability:    {torch.cuda.get_device_capability(0)}")
    say(f"arch list:     {arch}")
    say(f"cuDNN:         {torch.backends.cudnn.version()}")
    say(f"bf16 support:  {torch.cuda.is_bf16_supported()}")
    say(f"fp16 support:  True (matmul below)")
    say(f"total VRAM:    {props.total_memory / 2**30:.2f} GiB")
    if not any(a in ("sm_120", "compute_120") for a in arch):
        say("WARNING: sm_120/compute_120 not in arch list")

    say("\nTimings (ms; median [min, max], 2 warm-up + 5 runs):")
    for dt in (torch.float16, torch.bfloat16):
        a = torch.randn(4096, 4096, device=dev, dtype=dt)
        b = torch.randn(4096, 4096, device=dev, dtype=dt)
        m, lo, hi = timed(lambda: a @ b)
        tf = 2 * 4096**3 / (m / 1000) / 1e12
        say(f"matmul 4096x4096 {str(dt):15s} {m:8.3f} [{lo:.3f}, {hi:.3f}]  ~{tf:.1f} TFLOPS")
        assert torch.isfinite((a @ b).float()).all()

    x = torch.randn(1, 64, 1024, 1024, device=dev)
    conv = torch.nn.Conv2d(64, 64, 3, padding=1).to(dev)
    with torch.no_grad():
        m, lo, hi = timed(lambda: conv(x))
    say(f"conv3x3 1x64x1024x1024 fp32    {m:8.3f} [{lo:.3f}, {hi:.3f}]")

    n, hw = 10_000_000, 1024 * 1024
    idx = torch.randint(0, hw, (n,), device=dev)
    src = torch.rand(n, device=dev)
    out = torch.zeros(hw, device=dev)
    m, lo, hi = timed(lambda: out.scatter_reduce_(0, idx, src, "amax", include_self=True))
    say(f"scatter_reduce amax 10M        {m:8.3f} [{lo:.3f}, {hi:.3f}]")
    out2 = torch.zeros(hw, device=dev)
    m, lo, hi = timed(lambda: out2.index_add_(0, idx, src))
    say(f"index_add_ 10M                 {m:8.3f} [{lo:.3f}, {hi:.3f}]")
    assert torch.isfinite(out2).all() and out.max() > 0

    say(f"\npeak VRAM allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    say("ALL OK")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
