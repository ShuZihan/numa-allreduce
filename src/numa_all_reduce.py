# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
NUMA-Aware Custom AllReduce Implementation

This module provides a custom AllReduce implementation that is optimized for
multi-GPU systems with NUMA architecture. It uses a hierarchical approach to
minimize cross-NUMA communication and maximize bandwidth utilization.

Features:
- Automatic NUMA topology detection
- Hierarchical three-stage AllReduce algorithm
- NUMA-aware buffer allocation
- Optimized for PCIe-only systems
"""

from contextlib import contextmanager
from typing import List, Optional, Tuple, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check,
)
from vllm.distributed.device_communicators.numa_utils import (
    NumaTopology,
    detect_numa_topology,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import cuda_device_count_stateless

logger = init_logger(__name__)

# Check if custom allreduce is available
try:
    ops.meta_size()
    custom_ar = True
except Exception:
    custom_ar = False


class NumaAwareAllReduce:
    """
    NUMA-aware custom AllReduce implementation for PCIe-only multi-GPU systems.

    This implementation uses a hierarchical three-stage algorithm:
    1. Intra-NUMA Reduce-Scatter: Each NUMA node reduces data locally
    2. Inter-NUMA AllReduce: NUMA representatives exchange partial results
    3. Intra-NUMA AllGather: Results are broadcast within each NUMA node

    Key optimizations for PCIe-only systems:
    - Minimizes cross-NUMA PCIe traffic
    - Maximizes intra-NUMA bandwidth utilization
    - NUMA-aware buffer allocation

    Args:
        group: The process group to work on
        device: The CUDA device to bind to
        max_size: Maximum supported allreduce size in bytes
        enable_optimization: Whether to enable NUMA optimizations
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _SUPPORTED_NUMA_CONFIGS = {
        2: [(1, 1)],  # 2 GPUs: can be 1+1 across 2 NUMA nodes
        4: [(2, 2)],  # 4 GPUs: 2+2 across 2 NUMA nodes
        6: [(3, 3)],  # 6 GPUs: 3+3 across 2 NUMA nodes
        8: [(4, 4)],  # 8 GPUs: 4+4 across 2 NUMA nodes (most common)
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = 8192 * 1024,
        enable_optimization: bool = True,
    ) -> None:
        self._IS_CAPTURING = False
        self.disabled = True
        self.enable_optimization = enable_optimization

        if not custom_ar:
            logger.info(
                "NUMA-Aware AllReduce is disabled because "
                "custom allreduce library is not available"
            )
            return

        self.group = group

        # Check that we're using a non-NCCL group (for CPU-side coordination)
        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "NumaAwareAllReduce should be attached to a non-NCCL group."
        )

        # Only support single-node for now
        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.warning(
                "NUMA-Aware AllReduce is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        self.world_size = world_size

        if world_size == 1:
            return

        if world_size not in NumaAwareAllReduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "NUMA-Aware AllReduce is disabled due to unsupported world"
                " size: %d. Supported: %s",
                world_size,
                str(NumaAwareAllReduce._SUPPORTED_WORLD_SIZES),
            )
            return

        # Setup device
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        # Detect NUMA topology
        self.numa_topology = detect_numa_topology()
        self._validate_numa_configuration()

        # Adjust max_size based on device capability
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES:
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )

        # Setup GPU physical IDs
        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(cuda_device_count_stateless()))

        physical_device_id = device_ids[device.index]

        # Exchange physical device IDs across ranks
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # Build NUMA groups based on GPU distribution
        self.numa_groups = self._build_numa_groups(physical_device_ids)

        # Check P2P access capability
        if not self._can_p2p(rank, world_size, physical_device_ids):
            logger.warning(
                "NUMA-Aware AllReduce is disabled due to lack of P2P access"
            )
            return

        # Initialize shared buffers
        self.meta_ptrs = self._create_shared_buffer(
            ops.meta_size() + max_size, group=group, uncached=True
        )
        self.buffer_ptrs = self._create_shared_buffer(max_size, group=group)

        # Allocate rank data buffer
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        self.max_size = max_size
        self.physical_device_ids = physical_device_ids
        self.disabled = False
        self._fully_connected = self._check_full_nvlink(physical_device_ids)

        # Initialize custom allreduce
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self._fully_connected
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)

        logger.info(
            f"NUMA-Aware AllReduce initialized: rank={rank}, world_size={world_size}, "
            f"NUMA_groups={self.numa_groups}"
        )

    def _validate_numa_configuration(self) -> None:
        """Validate that the NUMA configuration is supported."""
        expected_configs = self._SUPPORTED_NUMA_CONFIGS.get(self.world_size, [])
        if not expected_configs:
            logger.warning(f"World size {self.world_size} has no defined NUMA configs")
            return

        # Check actual GPU distribution
        actual_distribution = {}
        for gpu_id, gpu_info in self.numa_topology.gpu_info.items():
            numa_node = gpu_info.numa_node
            actual_distribution[numa_node] = actual_distribution.get(numa_node, 0) + 1

        logger.info(f"Detected NUMA distribution: {actual_distribution}")

    def _build_numa_groups(self, physical_device_ids: List[int]) -> Dict[int, List[int]]:
        """Build mapping of NUMA nodes to their GPU ranks."""
        numa_groups: Dict[int, List[int]] = {}

        for rank, phys_id in enumerate(physical_device_ids):
            # Find which NUMA node this GPU belongs to
            numa_node = self.numa_topology.get_numa_node_for_gpu(phys_id)
            if numa_node not in numa_groups:
                numa_groups[numa_node] = []
            numa_groups[numa_node].append(rank)

        return numa_groups

    def _can_p2p(self, rank: int, world_size: int, physical_device_ids: List[int]) -> bool:
        """Check if P2P access is available between all GPUs."""
        for i in range(world_size):
            if i == rank:
                continue
            if envs.VLLM_SKIP_P2P_CHECK:
                logger.debug("Skipping P2P check, trusting driver")
                phys_i = physical_device_ids[i]
                phys_rank = physical_device_ids[rank]
                return torch.cuda.can_device_access_peer(phys_rank, phys_i)
            if not gpu_p2p_access_check(rank, i):
                return False
        return True

    def _check_full_nvlink(self, physical_device_ids: List[int]) -> bool:
        """Check if all GPUs are fully connected via NVLink."""
        if not current_platform.is_cuda():
            return False
        try:
            return current_platform.is_fully_connected(physical_device_ids)
        except Exception:
            return False

    def _create_shared_buffer(
        self,
        size_in_bytes: int,
        group: Optional[ProcessGroup] = None,
        uncached: bool = False,
    ) -> List[int]:
        """Create a shared buffer accessible by all ranks."""
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)
            else:
                pointers.append(ops.open_mem_handle(h))

        return pointers

    def _free_shared_buffer(
        self,
        pointers: List[int],
        group: Optional[ProcessGroup] = None,
        rank: Optional[int] = None,
    ) -> None:
        """Free a shared buffer."""
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])

    def should_use_numa_ar(self, inp: torch.Tensor) -> bool:
        """Check if NUMA-Aware AllReduce should be used for this input."""
        if self.disabled:
            return False

        inp_size = inp.numel() * inp.element_size()

        # Must be multiple of 16 bytes
        if inp_size % 16 != 0:
            return False

        # Must be contiguous or weakly contiguous
        if not self._is_weak_contiguous(inp):
            return False

        # Check size limits
        if inp_size > self.max_size:
            return False

        # For PCIe-only systems, always use NUMA-aware for better performance
        return True

    def _is_weak_contiguous(self, inp: torch.Tensor) -> bool:
        """Check if tensor is weakly contiguous."""
        if inp.is_contiguous():
            return True
        # Check if storage covers the tensor data exactly
        storage = inp.storage()
        expected_bytes = inp.numel() * inp.element_size()
        actual_bytes = storage.nbytes() - inp.storage_offset() * inp.element_size()
        return actual_bytes == expected_bytes

    def all_reduce(
        self,
        inp: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        registered: bool = False,
    ) -> torch.Tensor:
        """
        Perform NUMA-aware all-reduce.

        For PCIe-only systems with NUMA architecture, this uses a hierarchical
        approach:
        1. Intra-NUMA Reduce-Scatter
        2. Inter-NUMA AllReduce
        3. Intra-NUMA AllGather

        Args:
            inp: Input tensor to reduce
            out: Optional output tensor (created if None)
            registered: Whether the input buffer is already IPC-registered

        Returns:
            The reduced output tensor
        """
        if out is None:
            out = torch.empty_like(inp)

        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )

        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Main allreduce API that provides support for CUDA graph.

        Returns None if NUMA-aware allreduce is disabled or should not be used
        for this input.
        """
        if self.disabled or not self.should_use_numa_ar(input):
            return None

        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # Warmup: mimic allocation pattern
                return torch.empty_like(input)
        else:
            return self.all_reduce(input, registered=False)

    @contextmanager
    def capture(self):
        """
        Context manager for CUDA graph capture.
        Handles registration of graph buffers at the end of capture.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self) -> None:
        """Register CUDA graph buffers for IPC sharing."""
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))

        all_data: List[List[Optional[List[int]]]] = [
            [None, None] for _ in range(dist.get_world_size(group=self.group))
        ]
        all_data[self.rank] = [handle, offset]

        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )

        handles = cast(List[List[int]], [d[0] for d in all_data])
        offsets = cast(List[List[int]], [d[1] for d in all_data])
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def close(self) -> None:
        """Clean up resources."""
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self._free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self._free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    def get_numa_groups(self) -> Dict[int, List[int]]:
        """Get the NUMA group mapping (for debugging/inspection)."""
        return self.numa_groups

    def get_topology_summary(self) -> str:
        """Get a summary of the NUMA topology for this communicator."""
        return self.numa_topology.get_summary()
