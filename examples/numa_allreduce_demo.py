#!/usr/bin/env python3
"""
NUMA-Aware AllReduce Demo

This script demonstrates the usage of NUMA-aware custom AllReduce
optimized for PCIe-only multi-GPU systems.

Usage:
    python examples/numa_allreduce_demo.py --world_size 4

Requirements:
- Multi-GPU system with PCIe-connected GPUs
- PyTorch with CUDA support
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


def run_demo(rank: int, world_size: int, test_config: dict):
    """Run the NUMA AllReduce demo on a single GPU."""
    # Initialize distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    try:
        # Only rank 0 prints topology info
        if rank == 0:
            print("\n" + "=" * 70)
            print("NUMA-Aware AllReduce Demo")
            print("=" * 70)

            # Detect and print NUMA topology
            topology = NumaTopology.detect()
            topology.print_topology()
            print("=" * 70 + "\n")

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

        # Run correctness test
        if test_config.get('test_correctness', True):
            _run_correctness_test(rank, world_size, na_ar, test_config)

        # Run performance benchmark
        if test_config.get('test_performance', True) and not na_ar.disabled:
            _run_performance_test(rank, world_size, na_ar, test_config)

        dist.barrier()

        if rank == 0:
            print("\n" + "=" * 70)
            print("Demo completed successfully!")
            print("=" * 70 + "\n")

    finally:
        dist.destroy_process_group()


def _run_correctness_test(rank, world_size, na_ar, test_config):
    """Run correctness test comparing NUMA AR to NCCL."""
    if rank == 0:
        print("-" * 70)
        print("Running Correctness Test")
        print("-" * 70)

    # Test different tensor sizes
    test_sizes = [
        (1024, 1024),      # 4 MB (float32)
        (4096, 4096),      # 64 MB (float32)
    ]

    for shape in test_sizes:
        # Create test tensor
        torch.manual_seed(42 + rank)
        local_tensor = torch.randn(shape, dtype=torch.float32, device='cuda')

        # Compute expected result with NCCL
        expected = local_tensor.clone()
        dist.all_reduce(expected)

        # Compute result with NUMA AR
        if not na_ar.disabled:
            test_input = local_tensor.clone()
            result = na_ar.custom_all_reduce(test_input)

            if result is not None:
                # Check correctness
                max_diff = (result - expected).abs().max().item()
                is_correct = max_diff < 1e-3

                if rank == 0:
                    status = "PASS" if is_correct else "FAIL"
                    size_mb = shape[0] * shape[1] * 4 / (1024 * 1024)
                    print(f"  Shape {shape} ({size_mb:.1f} MB): {status} "
                          f"(max_diff={max_diff:.2e})")

                assert is_correct, f"Correctness check failed for shape {shape}"

        dist.barrier()

    if rank == 0:
        print("  All correctness tests passed!")
        print()


def _run_performance_test(rank, world_size, na_ar, test_config):
    """Run performance benchmark comparing NUMA AR to NCCL."""
    if rank == 0:
        print("-" * 70)
        print("Running Performance Benchmark")
        print("-" * 70)

    # Test different tensor sizes
    test_sizes_mb = [64, 256, 1024]
    iterations = 100
    warmup = 20

    for size_mb in test_sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        numel = size_bytes // 4  # float32

        # Create test tensor
        tensor = torch.randn(numel, dtype=torch.float32, device='cuda')

        # Benchmark NCCL
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
            nccl_times.append((end - start) * 1000)  # ms

        nccl_avg = sum(nccl_times) / len(nccl_times)
        nccl_std = (sum((t - nccl_avg) ** 2 for t in nccl_times) / len(nccl_times)) ** 0.5

        # Benchmark NUMA-Aware AllReduce
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
                numa_times.append((end - start) * 1000)  # ms

        if numa_times:
            numa_avg = sum(numa_times) / len(numa_times)
            numa_std = (sum((t - numa_avg) ** 2 for t in numa_times) / len(numa_times)) ** 0.5
            speedup = nccl_avg / numa_avg if numa_avg > 0 else 1.0
        else:
            numa_avg = float('nan')
            numa_std = float('nan')
            speedup = 0.0

        # Print results
        if rank == 0:
            print(f"\n  Size: {size_mb} MB")
            print(f"  {'Implementation':<20} {'Avg (ms)':<15} {'Std (ms)':<15}")
            print(f"  {'-'*50}")
            print(f"  {'NCCL':<20} {nccl_avg:<15.3f} {nccl_std:<15.3f}")
            print(f"  {'NUMA-Aware':<20} {numa_avg:<15.3f} {numa_std:<15.3f}")
            print(f"  Speedup: {speedup:.2f}x")

        dist.barrier()

    if rank == 0:
        print("\n  Performance benchmark completed!")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='NUMA-Aware AllReduce Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with 4 GPUs
    python examples/numa_allreduce_demo.py --world_size 4

    # Run with correctness test only
    python examples/numa_allreduce_demo.py --world_size 4 --no-performance

    # Run with performance test only
    python examples/numa_allreduce_demo.py --world_size 4 --no-correctness
        """
    )

    parser.add_argument(
        '--world_size',
        type=int,
        default=min(torch.cuda.device_count(), 4),
        help='Number of GPUs to use (default: min(4, num_gpus))'
    )
    parser.add_argument(
        '--no-correctness',
        action='store_true',
        help='Skip correctness test'
    )
    parser.add_argument(
        '--no-performance',
        action='store_true',
        help='Skip performance test'
    )

    args = parser.parse_args()

    # Validate world_size
    if args.world_size > torch.cuda.device_count():
        print(f"Error: Requested {args.world_size} GPUs but only "
              f"{torch.cuda.device_count()} available")
        sys.exit(1)

    if args.world_size < 2:
        print("Error: Need at least 2 GPUs for AllReduce demo")
        sys.exit(1)

    # Prepare test configuration
    test_config = {
        'test_correctness': not args.no_correctness,
        'test_performance': not args.no_performance,
    }

    print(f"\nStarting NUMA-Aware AllReduce Demo")
    print(f"World Size: {args.world_size} GPUs")
    print(f"Tests: Correctness={test_config['test_correctness']}, "
          f"Performance={test_config['test_performance']}")
    print()

    # Run the demo
    mp.spawn(
        run_demo,
        args=(args.world_size, test_config),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == '__main__':
    main()
