#!/usr/bin/env python3
"""
NSys / NCU 性能采集脚本

专门用于 NVIDIA Nsight Systems (NSys) 和 Nsight Compute (NCU) 的性能采集。
对比 NCCL Ring AllReduce 和 NUMA AllReduce 的性能。

Usage:
    # 使用 NSys 采集（系统级分析）
    python examples/profile_nsys_ncu.py --world_size 4 --mode nsys

    # 使用 NCU 采集（Kernel级分析）
    python examples/profile_nsys_ncu.py --world_size 4 --mode ncu

    # 仅运行基准测试（不使用 profiler）
    python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark
"""

import argparse
import os
import sys
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from numa_utils import NumaTopology


def profile_worker(rank: int, world_size: int, config: dict):
    """Worker process for profiling."""
    # Initialize distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29600'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    try:
        # Only rank 0 prints info
        if rank == 0:
            print("\n" + "=" * 70)
            print("NUMA AllReduce Profiling")
            print(f"Mode: {config['mode']}")
            print(f"World Size: {world_size}")
            print("=" * 70 + "\n")

            # Print NUMA topology
            topology = NumaTopology.detect()
            topology.print_topology()
            print()

        # Synchronize before starting
        dist.barrier()

        # Import here to avoid issues with process initialization
        from numa_all_reduce import NumaAwareAllReduce

        # Initialize NUMA-aware AllReduce
        na_ar = NumaAwareAllReduce(
            group=dist.group.WORLD,
            device=rank,
        )

        if rank == 0:
            print(f"NUMA-Aware AllReduce initialized: disabled={na_ar.disabled}")
            if not na_ar.disabled:
                print(f"NUMA groups: {na_ar.get_numa_groups()}")
            print()

        # Run profiling
        if config['mode'] == 'benchmark':
            _run_benchmark(rank, world_size, na_ar, config)
        elif config['mode'] == 'nsys':
            _run_nsys_profile(rank, world_size, na_ar, config)
        elif config['mode'] == 'ncu':
            _run_ncu_profile(rank, world_size, na_ar, config)

        dist.barrier()

        if rank == 0:
            print("\n" + "=" * 70)
            print("Profiling completed!")
            print("=" * 70 + "\n")

    finally:
        dist.destroy_process_group()


def _run_benchmark(rank, world_size, na_ar, config):
    """Run full benchmark without profiler."""
    if rank == 0:
        print("-" * 70)
        print("Running Benchmark")
        print("-" * 70)

    size_mb = config['size_mb']
    iterations = config['iterations']
    warmup = config['warmup']
    dtype_name = config['dtype']
    dtype = getattr(torch, dtype_name)

    size_bytes = size_mb * 1024 * 1024
    elem_size = 4 if dtype_name == 'float32' else 2
    numel = size_bytes // elem_size

    # Create test tensor
    tensor = torch.randn(numel, dtype=dtype, device='cuda')

    # ========== Benchmark NCCL ==========
    if rank == 0:
        print(f"\nBenchmarking NCCL ({dtype_name}, {size_mb} MB)...")

    nccl_times = []

    # Warmup
    for _ in range(warmup):
        test_tensor = tensor.clone()
        dist.all_reduce(test_tensor)
    torch.cuda.synchronize()

    # Benchmark
    for _ in range(iterations):
        test_tensor = tensor.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(test_tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()
        nccl_times.append((end - start) * 1000)

    nccl_avg = sum(nccl_times) / len(nccl_times)
    nccl_std = (sum((t - nccl_avg) ** 2 for t in nccl_times) / len(nccl_times)) ** 0.5

    # ========== Benchmark NUMA-Aware AllReduce ==========
    numa_avg = float('nan')
    numa_std = float('nan')
    speedup = 0.0

    if not na_ar.disabled:
        if rank == 0:
            print(f"Benchmarking NUMA-Aware ({dtype_name}, {size_mb} MB)...")

        numa_times = []

        # Warmup
        for _ in range(warmup):
            test_tensor = tensor.clone()
            _ = na_ar.custom_all_reduce(test_tensor)
        torch.cuda.synchronize()

        # Benchmark
        for _ in range(iterations):
            test_tensor = tensor.clone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = na_ar.custom_all_reduce(test_tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            if result is not None:
                numa_times.append((end - start) * 1000)

        if numa_times:
            numa_avg = sum(numa_times) / len(numa_times)
            numa_std = (sum((t - numa_avg) ** 2 for t in numa_times) / len(numa_times)) ** 0.5
            speedup = nccl_avg / numa_avg if numa_avg > 0 else 1.0

    # Print results
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"SUMMARY - {size_mb} MB {dtype_name}")
        print(f"{'='*70}")
        print(f"{'Implementation':<20} {'Avg (ms)':<15} {'Std (ms)':<15} {'Bandwidth (GB/s)':<18}")
        print(f"{'-'*68}")

        nccl_bw = (size_bytes * 2 / 1e9) / (nccl_avg / 1000)
        print(f"{'NCCL':<20} {nccl_avg:<15.3f} {nccl_std:<15.3f} {nccl_bw:<18.2f}")

        if not na_ar.disabled and numa_times:
            numa_bw = (size_bytes * 2 / 1e9) / (numa_avg / 1000)
            print(f"{'NUMA-Aware':<20} {numa_avg:<15.3f} {numa_std:<15.3f} {numa_bw:<18.2f}")
            print(f"\nSpeedup: {speedup:.2f}x")
        else:
            print(f"{'NUMA-Aware':<20} {'DISABLED':<15}")
        print(f"{'='*70}\n")


def _run_nsys_profile(rank, world_size, na_ar, config):
    """Run profiling with Nsight Systems markers."""
    if rank == 0:
        print("-" * 70)
        print("Running NSys Profiling")
        print("-" * 70)
        print("\nTo capture with NSys, run:")
        print("  nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \\")
        print("    -o numa_allreduce_profile python examples/profile_nsys_ncu.py --world_size 4 --mode nsys")
        print()

    size_mb = config['size_mb']
    iterations = config['iterations']
    warmup = config['warmup']
    dtype_name = config['dtype']
    dtype = getattr(torch, dtype_name)

    size_bytes = size_mb * 1024 * 1024
    elem_size = 4 if dtype_name == 'float32' else 2
    numel = size_bytes // elem_size

    # Create test tensor
    tensor = torch.randn(numel, dtype=dtype, device='cuda')

    # Warmup
    for _ in range(warmup):
        test_tensor = tensor.clone()
        dist.all_reduce(test_tensor)
        if not na_ar.disabled:
            _ = na_ar.custom_all_reduce(test_tensor.clone())
    torch.cuda.synchronize()

    # Try to use torch.cuda.nvtx if available
    try:
        from torch.cuda import nvtx
        has_nvtx = True
    except ImportError:
        has_nvtx = False
        if rank == 0:
            print("Note: torch.cuda.nvtx not available, skipping NVTX markers")

    # ========== Profile NCCL ==========
    if rank == 0:
        print(f"\nProfiling NCCL ({dtype_name}, {size_mb} MB)...")

    if has_nvtx:
        nvtx.range_push("NCCL_AllReduce")

    for i in range(iterations):
        if has_nvtx:
            nvtx.range_push(f"NCCL_Iter_{i}")
        test_tensor = tensor.clone()
        dist.all_reduce(test_tensor)
        if has_nvtx:
            nvtx.range_pop()

    if has_nvtx:
        nvtx.range_pop()

    torch.cuda.synchronize()

    # ========== Profile NUMA-Aware AllReduce ==========
    if not na_ar.disabled:
        if rank == 0:
            print(f"Profiling NUMA-Aware ({dtype_name}, {size_mb} MB)...")

        if has_nvtx:
            nvtx.range_push("NUMA_AllReduce")

        for i in range(iterations):
            if has_nvtx:
                nvtx.range_push(f"NUMA_Iter_{i}")
            test_tensor = tensor.clone()
            _ = na_ar.custom_all_reduce(test_tensor)
            if has_nvtx:
                nvtx.range_pop()

        if has_nvtx:
            nvtx.range_pop()

        torch.cuda.synchronize()

    if rank == 0:
        print("\nProfiling phase completed!")
        print("View results with: nsys-ui numa_allreduce_profile.qdrep")


def _run_ncu_profile(rank, world_size, na_ar, config):
    """Run profiling for Nsight Compute."""
    if rank == 0:
        print("-" * 70)
        print("Running NCU Profiling")
        print("-" * 70)
        print("\nNote: NCU is typically run from the command line, not within Python.")
        print("\nTo profile specific kernels with NCU, run:")
        print("  # Profile NCCL kernels")
        print("  ncu --set full -o nccl_profile python examples/profile_nsys_ncu.py --world_size 4 --mode ncu")
        print()
        print("  # Profile custom NUMA kernels")
        print("  ncu --set full -o numa_profile python examples/profile_nsys_ncu.py --world_size 4 --mode ncu")
        print()

    size_mb = config['size_mb']
    iterations = config['iterations']
    warmup = config['warmup']
    dtype_name = config['dtype']
    dtype = getattr(torch, dtype_name)

    size_bytes = size_mb * 1024 * 1024
    elem_size = 4 if dtype_name == 'float32' else 2
    numel = size_bytes // elem_size

    # Create test tensor
    tensor = torch.randn(numel, dtype=dtype, device='cuda')

    # Warmup
    for _ in range(warmup):
        test_tensor = tensor.clone()
        dist.all_reduce(test_tensor)
        if not na_ar.disabled:
            _ = na_ar.custom_all_reduce(test_tensor.clone())
    torch.cuda.synchronize()

    # ========== Profile NCCL ==========
    if rank == 0:
        print(f"\nRunning NCCL iterations for NCU ({dtype_name}, {size_mb} MB)...")

    for _ in range(iterations):
        test_tensor = tensor.clone()
        dist.all_reduce(test_tensor)

    torch.cuda.synchronize()

    # ========== Profile NUMA-Aware AllReduce ==========
    if not na_ar.disabled:
        if rank == 0:
            print(f"Running NUMA-Aware iterations for NCU ({dtype_name}, {size_mb} MB)...")

        for _ in range(iterations):
            test_tensor = tensor.clone()
            _ = na_ar.custom_all_reduce(test_tensor)

        torch.cuda.synchronize()

    if rank == 0:
        print("\nNCU profiling phase completed!")


def main():
    parser = argparse.ArgumentParser(
        description='NUMA AllReduce NSys/NCU Profiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run benchmark only
    python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark

    # Run with NSys profiling mode (add markers)
    python examples/profile_nsys_ncu.py --world_size 4 --mode nsys

    # Run with NCU profiling mode
    python examples/profile_nsys_ncu.py --world_size 4 --mode ncu

    # Custom size and iterations
    python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark --size_mb 256 --iterations 200
        """
    )

    parser.add_argument(
        '--world_size',
        type=int,
        default=min(torch.cuda.device_count(), 4),
        help='Number of GPUs to use (default: min(4, num_gpus))'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='benchmark',
        choices=['benchmark', 'nsys', 'ncu'],
        help='Profiling mode: benchmark, nsys, or ncu'
    )
    parser.add_argument(
        '--size_mb',
        type=int,
        default=256,
        help='Tensor size in MB (default: 256)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type (default: float32)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of iterations (default: 100)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=20,
        help='Number of warmup iterations (default: 20)'
    )

    args = parser.parse_args()

    # Validate world_size
    if args.world_size > torch.cuda.device_count():
        print(f"Error: Requested {args.world_size} GPUs but only "
              f"{torch.cuda.device_count()} available")
        sys.exit(1)

    if args.world_size < 2:
        print("Error: Need at least 2 GPUs for AllReduce profiling")
        sys.exit(1)

    # Prepare config
    config = {
        'mode': args.mode,
        'size_mb': args.size_mb,
        'dtype': args.dtype,
        'iterations': args.iterations,
        'warmup': args.warmup,
    }

    # Run profiling
    mp.spawn(
        profile_worker,
        args=(args.world_size, config),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == '__main__':
    main()
