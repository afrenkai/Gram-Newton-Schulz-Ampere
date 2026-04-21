import argparse
import sys
import time
from datetime import datetime
from typing import Callable
import torch
from triton.testing import do_bench
from newton_schultz import NewtonSchulz

# https://x.com/YouJiacheng/status/1905861218138804534
YOU_COEFFICIENTS = [
    [4.0848, -6.8946, 2.9270],
    [3.9505, -6.3029, 2.6377],
    [3.7418, -5.5913, 2.3037],
    [2.8769, -3.1427, 1.2046],
    [2.8366, -3.0525, 1.2012],
]


def benchmark(
    callable_fn: Callable, X: torch.Tensor, warmup_iter: int, repeat_iter: int
):
    timing_ms = do_bench(lambda: callable_fn(X), warmup_iter, rep=repeat_iter)
    print(f"timing_ms: {timing_ms:8.4f} milliseconds")
    return timing_ms


def main(args):
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    compute_capability = capability[0] * 10 + capability[1]
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"Compute Capability: {capability[0]}.{capability[1]} (SM{compute_capability})"
    )
    print(f"Batch size: {args.batch_size}")
    print(f"Input dtype: bfloat16")
    print(f"Warmup iterations: {args.warmup_iter}")
    print(f"Benchmark iterations: {args.repeat_iter}")

    can_use_kernels = compute_capability >= 90

    if can_use_kernels:
        print("Custom kernels available (H100/B200)")
    else:
        print(
            f"Custom kernels not available (requires SM90+, found SM{compute_capability})"
        )
        print("Will only benchmark PyTorch implementations")

    torch_dtype = torch.bfloat16

    M, N = args.rows, args.cols
    print(f"Shape: {M}x{N} | Batch size: {args.batch_size}")

    X = torch.randn(args.batch_size, M, N, dtype=torch_dtype, device="cuda")

    standard_torch = NewtonSchulz(
        eps=1e-9, coeff=YOU_COEFFICIENTS, use_gram=False, gns_reset_iters=None
    )

    _ = standard_torch(X)
    torch.cuda.synchronize()
    time.sleep(1.0)

    timing_standard_torch = benchmark(
        standard_torch,
        X,
        warmup_iter=args.warmup_iter,
        repeat_iter=args.repeat_iter,
    )

    torch.cuda.synchronize()
    time.sleep(1.0)

    timing_standard_kernels = None
    if can_use_kernels:
        print("\n[2] Standard Newton-Schulz (Kernels)")
        standard_kernels = NewtonSchulz(
            eps=1e-7,
            coeff=YOU_COEFFICIENTS,
        )

        _ = standard_kernels(X)
        torch.cuda.synchronize()
        time.sleep(1.0)

        timing_standard_kernels = benchmark(
            standard_kernels,
            X,
            warmup_iter=args.warmup_iter,
            repeat_iter=args.repeat_iter,
        )
        torch.cuda.synchronize()
        time.sleep(1.0)

    gram_torch = NewtonSchulz(
        eps=1e-9,
        use_gram=True,
        coeff=YOU_COEFFICIENTS,
        gns_reset_iters=[2],
    )

    _ = gram_torch(X)
    torch.cuda.synchronize()
    time.sleep(1.0)

    timing_gram_torch = benchmark(
        gram_torch,
        X,
        warmup_iter=args.warmup_iter,
        repeat_iter=args.repeat_iter,
    )
    torch.cuda.synchronize()
    time.sleep(1.0)
    #
    # timing_gram_kernels = None
    # if can_use_kernels:
    #     print("\n[4] Gram Newton-Schulz (Kernels)")
    #     gram_kernels = GramNewtonSchulz(
    #         ns_epsilon=1e-7,
    #         ns_use_kernels=True,
    #         ns_coefficients=YOU_COEFFICIENTS,
    #         gram_newton_schulz_reset_iterations=[2],
    #     )
    #
    #     _ = gram_kernels(X)
    #     torch.cuda.synchronize()
    #     time.sleep(1.0)
    #
    #     timing_gram_kernels = benchmark_ns_variant(
    #         gram_kernels,
    #         X,
    #         warmup=args.warmup,
    #         repeats=args.repeats,
    #         desc="Gram Newton-Schulz (Kernels)",
    #     )
    #     torch.cuda.synchronize()
    #     time.sleep(1.0)

    print(f"{'Standard Newton-Schulz (PyTorch)':<50} | {timing_standard_torch:8.4f}")
    if can_use_kernels:
        print(
            f"{'Standard Newton-Schulz (Kernels)':<50} | {timing_standard_kernels:8.4f}"
        )
    print(f"{'Gram Newton-Schulz (PyTorch)':<50} | {timing_gram_torch:8.4f}")
    # if can_use_kernels:
    #     print(f"{'Gram Newton-Schulz (Kernels)':<50} | {timing_gram_kernels:12.3f}")
    #

    if args.profile:
        if args.profile_trace:
            trace_filename = args.profile_trace
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_filename = f"ns_profile_{timestamp}.json"

        print("Running Profiler")
        print(f"Output trace: {trace_filename}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            _ = standard_torch(X)
            torch.cuda.synchronize()

            # if can_use_kernels:
            #     _ = standard_kernels(X)
            #     torch.cuda.synchronize()
            #
            # _ = gram_torch(X)
            # torch.cuda.synchronize()
            #
            # if can_use_kernels:
            #     _ = gram_kernels(X)
            #     torch.cuda.synchronize()

        prof.export_chrome_trace(trace_filename)
        print(f"Trace saved to: {trace_filename}")
        print(f"View at: chrome://tracing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup_iter", type=int, default=5)
    parser.add_argument("--repeat_iter", type=int, default=30)
    parser.add_argument(
        "--profile",
        action="store_true",
    )
    parser.add_argument("--profile-trace", type=str, default=None)

    args = parser.parse_args()
    main(args)
