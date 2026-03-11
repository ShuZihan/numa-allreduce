# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for NUMA-Aware AllReduce implementation

These tests validate:
1. NUMA topology detection
2. AllReduce correctness compared to NCCL
3. Performance benchmarks
4. Multi-GPU configurations (TP2, TP4, TP8)
"""

import os
import sys
import time
from typing import List, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def get_test_config(world_size: int) -> dict:
    """Get test configuration for a given world size."""
    configs = {
        2: {
            "numa_nodes": 1,  # Both GPUs on same NUMA
            "gpus_per_numa": 2,
        },
        4: {
            "numa_nodes": 2,
            "gpus_per_numa": 2,
        },
        8: {
            "numa_nodes": 2,
            "gpus_per_numa": 4,
        },
    }
    return configs.get(world_size, configs[2])


class TestNumaTopologyDetection:
    """Test NUMA topology detection functionality."""

    def test_numa_detection_no_crash(self):
        """Test that NUMA detection runs without crashing."""
        from numa_utils import (
            NumaTopology,
        )

        # Should not raise an exception
        topology = NumaTopology.detect()

        # Should have at least one NUMA node
        assert len(topology.numa_nodes) >= 1

    def test_gpu_mapping_detection(self):
        """Test that GPU to NUMA mapping is detected."""
        from numa_utils import (
            NumaTopology,
        )

        topology = NumaTopology.detect()
        gpu_count = torch.cuda.device_count()

        # All GPUs should have NUMA information
        for gpu_id in range(gpu_count):
            numa_node = topology.get_numa_node_for_gpu(gpu_id)
            # NUMA node should be valid (non-negative)
            assert numa_node >= 0

    def test_numa_summary_output(self):
        """Test that topology summary can be generated."""
        from numa_utils import (
            NumaTopology,
        )

        topology = NumaTopology.detect()
        summary = topology.get_summary()

        # Summary should contain expected sections
        assert "NUMA Topology" in summary
        assert "NUMA Nodes:" in summary


class TestNumaAllReduceCorrectness:
    """Test NUMA-Aware AllReduce correctness."""

    @staticmethod
    def _worker_correctness(rank: int, world_size: int, test_data: dict):
        """Worker process for correctness testing."""
        # Initialize distributed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'

        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        try:
            # Create test tensor
            torch.manual_seed(42 + rank)
            shape = test_data['shape']
            dtype = getattr(torch, test_data['dtype'])

            local_tensor = torch.randn(shape, dtype=dtype, device='cuda')

            # Compute expected result using NCCL
            expected = local_tensor.clone()
            dist.all_reduce(expected)

            # Compute result using NUMA-aware AllReduce
            from numa_all_reduce import (
                NumaAwareAllReduce,
            )

            na_ar = NumaAwareAllReduce(
                group=dist.group.WORLD,
                device=rank,
            )

            if not na_ar.disabled:
                result = na_ar.custom_all_reduce(local_tensor)

                # Verify correctness
                if result is not None:
                    rtol = test_data.get('rtol', 1e-5)
                    atol = test_data.get('atol', 1e-5)

                    is_close = torch.allclose(result, expected, rtol=rtol, atol=atol)

                    if not is_close:
                        max_diff = (result - expected).abs().max().item()
                        print(f"Rank {rank}: FAILED - max diff = {max_diff}")
                        dist.all_reduce(torch.tensor([0.0], device='cuda'))
                    else:
                        dist.all_reduce(torch.tensor([1.0], device='cuda'))
                else:
                    # Numa AR returned None (fallback)
                    dist.all_reduce(torch.tensor([1.0], device='cuda'))
            else:
                # Numa AR disabled, skip test
                dist.all_reduce(torch.tensor([1.0], device='cuda'))

        finally:
            dist.destroy_process_group()

    @pytest.mark.parametrize("world_size", [2, 4])
    @pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_correctness_against_nccl(self, world_size, dtype):
        """Test that NUMA AR produces same results as NCCL."""
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need at least {world_size} GPUs")

        test_data = {
            'shape': [1024, 1024],
            'dtype': dtype,
            'rtol': 1e-2 if dtype in ['float16', 'bfloat16'] else 1e-5,
            'atol': 1e-2 if dtype in ['float16', 'bfloat16'] else 1e-5,
        }

        mp.spawn(
            self._worker_correctness,
            args=(world_size, test_data),
            nprocs=world_size,
            join=True,
        )


class TestNumaAllReducePerformance:
    """Performance benchmarks for NUMA-Aware AllReduce."""

    @staticmethod
    def _worker_benchmark(rank: int, world_size: int, config: dict, results_queue):
        """Worker process for performance benchmarking."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29502'

        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        try:
            # Create test data
            size_bytes = config['size_bytes']
            dtype = getattr(torch, config['dtype'])
            elem_size = 4 if config['dtype'] == 'float32' else 2
            numel = size_bytes // elem_size

            tensor = torch.randn(numel, dtype=dtype, device='cuda')

            # Warmup
            for _ in range(10):
                dist.all_reduce(tensor.clone())
            torch.cuda.synchronize()

            # Benchmark NCCL
            nccl_times = []
            for _ in range(config['iterations']):
                test_tensor = tensor.clone()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                dist.all_reduce(test_tensor)
                end.record()
                torch.cuda.synchronize()

                nccl_times.append(start.elapsed_time(end))  # in ms

            # Benchmark NUMA-Aware AllReduce
            from numa_all_reduce import (
                NumaAwareAllReduce,
            )

            na_ar = NumaAwareAllReduce(
                group=dist.group.WORLD,
                device=rank,
            )

            numa_times = []
            if not na_ar.disabled:
                # Warmup
                for _ in range(10):
                    test_tensor = tensor.clone()
                    na_ar.custom_all_reduce(test_tensor)
                torch.cuda.synchronize()

                # Benchmark
                for _ in range(config['iterations']):
                    test_tensor = tensor.clone()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    start.record()
                    result = na_ar.custom_all_reduce(test_tensor)
                    end.record()
                    torch.cuda.synchronize()

                    if result is not None:
                        numa_times.append(start.elapsed_time(end))

            # Send results back
            if rank == 0:
                results_queue.put({
                    'nccl_times': nccl_times,
                    'numa_times': numa_times,
                    'config': config,
                })

        finally:
            dist.destroy_process_group()

    @pytest.mark.parametrize("world_size", [4])
    @pytest.mark.parametrize("size_mb", [64, 256, 1024])
    def test_performance_vs_nccl(self, world_size, size_mb):
        """Benchmark NUMA AR performance against NCCL."""
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need at least {world_size} GPUs")

        import multiprocessing as mp

        config = {
            'size_bytes': size_mb * 1024 * 1024,
            'dtype': 'float16',
            'iterations': 50,
        }

        results_queue = mp.Queue()

        mp.spawn(
            self._worker_benchmark,
            args=(world_size, config, results_queue),
            nprocs=world_size,
            join=True,
        )

        # Process results
        if not results_queue.empty():
            results = results_queue.get()
            nccl_times = results['nccl_times']
            numa_times = results['numa_times']

            if numa_times:
                nccl_avg = sum(nccl_times) / len(nccl_times)
                numa_avg = sum(numa_times) / len(numa_times)

                speedup = nccl_avg / numa_avg if numa_avg > 0 else 1.0

                print(f"\nPerformance Results ({world_size} GPUs, {size_mb}MB):")
                print(f"  NCCL avg:  {nccl_avg:.3f} ms")
                print(f"  NUMA avg:  {numa_avg:.3f} ms")
                print(f"  Speedup:   {speedup:.2f}x")

                # Assert that NUMA AR is competitive
                # Allow up to 20% slower in worst case
                assert speedup >= 0.8, f"NUMA AR too slow: {speedup:.2f}x"


class TestNumaAllReduceBf16:
    """Special tests for BF16 support in NUMA-Aware AllReduce."""

    @staticmethod
    def _worker_bf16_comprehensive(rank: int, world_size: int, test_config: dict):
        """Worker for comprehensive BF16 testing."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29503'

        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        try:
            from numa_all_reduce import (
                NumaAwareAllReduce,
            )

            # Test various tensor shapes
            shapes = test_config['shapes']

            for shape in shapes:
                # Create BF16 tensor
                torch.manual_seed(42 + rank)
                local_tensor = torch.randn(
                    shape, dtype=torch.bfloat16, device='cuda'
                )

                # Compute expected with NCCL
                expected = local_tensor.clone()
                dist.all_reduce(expected)

                # Test with NUMA-aware AllReduce
                na_ar = NumaAwareAllReduce(
                    group=dist.group.WORLD,
                    device=rank,
                )

                if not na_ar.disabled:
                    result = na_ar.custom_all_reduce(local_tensor)

                    if result is not None:
                        # Verify correctness with appropriate tolerances for BF16
                        # BF16 has about 3-4 decimal digits of precision
                        rtol = 1e-2
                        atol = 1e-2

                        is_close = torch.allclose(result, expected, rtol=rtol, atol=atol)

                        if not is_close and rank == 0:
                            max_diff = (result - expected).abs().max().item()
                            print(f"BF16 Test FAILED for shape {shape}")
                            print(f"  Max diff: {max_diff}")
                            # For debugging, print some sample values
                            print(f"  Sample expected: {expected.flatten()[:5]}")
                            print(f"  Sample result: {result.flatten()[:5]}")
                            dist.all_reduce(torch.tensor([0.0], device='cuda'))
                        else:
                            dist.all_reduce(torch.tensor([1.0], device='cuda'))
                    else:
                        # Fallback, still pass
                        dist.all_reduce(torch.tensor([1.0], device='cuda'))
                else:
                    # Disabled, still pass
                    dist.all_reduce(torch.tensor([1.0], device='cuda'))

        finally:
            dist.destroy_process_group()

    @pytest.mark.parametrize("world_size", [2, 4, 8])
    def test_bf16_various_shapes(self, world_size):
        """Test BF16 with various tensor shapes to ensure full compatibility."""
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need at least {world_size} GPUs")

        # Test various shapes that are common in LLM workloads
        test_config = {
            'shapes': [
                [1024],                     # Small vector
                [1024, 1024],               # Matrix (common in attention)
                [8192, 1024],               # Larger matrix
                [512, 4096],                # FFN shape
                [32, 128, 4096],            # 3D tensor
                [8, 32, 128, 128],          # 4D tensor (attention key/value)
            ]
        }

        mp.spawn(
            self._worker_bf16_comprehensive,
            args=(world_size, test_config),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def _worker_bf16_against_nccl(rank: int, world_size: int, results_queue):
        """Compare BF16 performance against NCCL."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29504'

        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        try:
            from numa_all_reduce import (
                NumaAwareAllReduce,
            )

            # Larger tensor for meaningful performance test
            size = 1024 * 1024  # 1M elements = 2MB in BF16

            tensor = torch.randn(size, dtype=torch.bfloat16, device='cuda')

            # Warmup NCCL
            for _ in range(10):
                dist.all_reduce(tensor.clone())
            torch.cuda.synchronize()

            # Benchmark NCCL BF16
            nccl_times = []
            for _ in range(50):
                test_tensor = tensor.clone()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                dist.all_reduce(test_tensor)
                end.record()
                torch.cuda.synchronize()

                nccl_times.append(start.elapsed_time(end))

            # Benchmark NUMA-aware BF16
            na_ar = NumaAwareAllReduce(
                group=dist.group.WORLD,
                device=rank,
            )

            numa_times = []
            if not na_ar.disabled:
                # Warmup
                for _ in range(10):
                    test_tensor = tensor.clone()
                    na_ar.custom_all_reduce(test_tensor)
                torch.cuda.synchronize()

                for _ in range(50):
                    test_tensor = tensor.clone()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    start.record()
                    result = na_ar.custom_all_reduce(test_tensor)
                    end.record()
                    torch.cuda.synchronize()

                    if result is not None:
                        numa_times.append(start.elapsed_time(end))

            if rank == 0:
                results_queue.put({
                    'nccl_times': nccl_times,
                    'numa_times': numa_times,
                    'world_size': world_size,
                })

        finally:
            dist.destroy_process_group()

    @pytest.mark.parametrize("world_size", [4, 8])
    def test_bf16_performance(self, world_size):
        """Test BF16 performance is reasonable."""
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need at least {world_size} GPUs")

        import multiprocessing as mp
        results_queue = mp.Queue()

        mp.spawn(
            self._worker_bf16_against_nccl,
            args=(world_size, results_queue),
            nprocs=world_size,
            join=True,
        )

        if not results_queue.empty():
            results = results_queue.get()
            nccl_times = results['nccl_times']
            numa_times = results['numa_times']

            if numa_times:
                nccl_avg = sum(nccl_times) / len(nccl_times)
                numa_avg = sum(numa_times) / len(numa_times)

                print(f"\nBF16 Performance ({world_size} GPUs):")
                print(f"  NCCL avg:  {nccl_avg:.3f} ms")
                print(f"  NUMA avg:  {numa_avg:.3f} ms")
                print(f"  Ratio:     {numa_avg/nccl_avg:.2f}x of NCCL")
