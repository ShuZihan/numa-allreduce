# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
NUMA Topology Detection and Management Utilities

This module provides utilities for detecting NUMA topology and GPU-NUMA mappings
to enable NUMA-aware optimizations for AllReduce operations.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class NumaNodeInfo:
    """Information about a NUMA node in the system."""
    node_id: int
    cpu_cores: List[int] = field(default_factory=list)
    gpu_ids: List[int] = field(default_factory=list)
    total_memory: int = 0  # in bytes
    free_memory: int = 0  # in bytes

    def __repr__(self) -> str:
        mem_gb = self.total_memory / (1024**3)
        return (f"NumaNodeInfo(node={self.node_id}, cpus={len(self.cpu_cores)}, "
                f"gpus={self.gpu_ids}, memory={mem_gb:.1f}GB)")


@dataclass
class GpuNumaInfo:
    """Information about a GPU and its NUMA affinity."""
    gpu_id: int
    numa_node: int
    pci_bus_id: str
    pci_domain: str = ""
    nvlink_peers: List[int] = field(default_factory=list)
    p2p_accessible: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"GpuNumaInfo(gpu={self.gpu_id}, numa={self.numa_node}, "
                f"pci={self.pci_bus_id}, nvlink={self.nvlink_peers})")


class NumaTopology:
    """
    Detects and manages NUMA topology information.

    This class provides methods to:
    1. Detect NUMA node configuration
    2. Map GPUs to NUMA nodes
    3. Detect NVLink and P2P connectivity
    4. Provide optimization recommendations

    Example usage:
        >>> topology = NumaTopology.detect()
        >>> print(topology.get_summary())
        >>> numa_for_gpu = topology.get_numa_node_for_gpu(0)
    """

    def __init__(self):
        self.numa_nodes: Dict[int, NumaNodeInfo] = {}
        self.gpu_info: Dict[int, GpuNumaInfo] = {}
        self._detect_numa_topology()
        self._detect_gpu_numa_mapping()

    @classmethod
    def detect(cls) -> "NumaTopology":
        """Factory method to detect and return NUMA topology."""
        return cls()

    def _detect_numa_topology(self) -> None:
        """Detect NUMA node information from system."""
        try:
            # Check if NUMA is available
            if not os.path.exists("/sys/devices/system/node"):
                logger.warning("NUMA not available on this system")
                self._create_default_numa()
                return

            # List all NUMA nodes
            node_dirs = [
                d for d in os.listdir("/sys/devices/system/node")
                if d.startswith("node") and d[4:].isdigit()
            ]

            for node_dir in sorted(node_dirs, key=lambda x: int(x[4:])):
                node_id = int(node_dir[4:])

                # Read CPU list
                cpu_cores = self._read_cpu_list(node_id)

                # Read memory info
                total_mem, free_mem = self._read_memory_info(node_id)

                self.numa_nodes[node_id] = NumaNodeInfo(
                    node_id=node_id,
                    cpu_cores=cpu_cores,
                    gpu_ids=[],
                    total_memory=total_mem,
                    free_memory=free_mem,
                )

            logger.info(f"Detected {len(self.numa_nodes)} NUMA nodes")

        except Exception as e:
            logger.warning(f"Failed to detect NUMA topology: {e}")
            self._create_default_numa()

    def _create_default_numa(self) -> None:
        """Create a default NUMA node for non-NUMA systems."""
        cpu_count = os.cpu_count() or 1
        self.numa_nodes[0] = NumaNodeInfo(
            node_id=0,
            cpu_cores=list(range(cpu_count)),
            gpu_ids=[],
            total_memory=0,
            free_memory=0,
        )

    def _read_cpu_list(self, node_id: int) -> List[int]:
        """Read CPU list for a NUMA node."""
        cpulist_path = f"/sys/devices/system/node/node{node_id}/cpulist"
        try:
            with open(cpulist_path, "r") as f:
                cpu_str = f.read().strip()
                return self._parse_cpulist(cpu_str)
        except Exception as e:
            logger.debug(f"Failed to read CPU list for node {node_id}: {e}")
            return []

    def _read_memory_info(self, node_id: int) -> Tuple[int, int]:
        """Read memory information for a NUMA node."""
        meminfo_path = f"/sys/devices/system/node/node{node_id}/meminfo"
        total_mem = 0
        free_mem = 0
        try:
            with open(meminfo_path, "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        match = re.search(r"(\d+)\s+kB", line)
                        if match:
                            total_mem = int(match.group(1)) * 1024
                    elif "MemFree" in line:
                        match = re.search(r"(\d+)\s+kB", line)
                        if match:
                            free_mem = int(match.group(1)) * 1024
        except Exception as e:
            logger.debug(f"Failed to read memory info for node {node_id}: {e}")
        return total_mem, free_mem

    def _parse_cpulist(self, cpulist_str: str) -> List[int]:
        """
        Parse CPU list string like '0-3,5,7-9' -> [0,1,2,3,5,7,8,9]
        """
        cores = []
        for part in cpulist_str.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start, end = map(int, part.split("-"))
                    cores.extend(range(start, end + 1))
                except ValueError:
                    continue
            else:
                try:
                    cores.append(int(part))
                except ValueError:
                    continue
        return cores

    def _detect_gpu_numa_mapping(self) -> None:
        """Detect GPU to NUMA node mapping."""
        if not torch.cuda.is_available():
            return

        try:
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get PCI info
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                pci_bus_id = f"{pci_info.bus:02x}:{pci_info.device:02x}.{pci_info.function}"

                # Determine NUMA node from PCI bus
                numa_node = self._get_numa_node_for_pci(pci_info.bus)

                # Detect NVLink peers
                nvlink_peers = []
                for j in range(device_count):
                    if i != j:
                        try:
                            peer_handle = pynvml.nvmlDeviceGetHandleByIndex(j)
                            p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                                handle, peer_handle,
                                pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                            )
                            if p2p_status == pynvml.NVML_P2P_STATUS_OK:
                                nvlink_peers.append(j)
                        except Exception:
                            pass

                self.gpu_info[i] = GpuNumaInfo(
                    gpu_id=i,
                    numa_node=numa_node,
                    pci_bus_id=pci_bus_id,
                    pci_domain=getattr(pci_info, 'domain', ''),
                    nvlink_peers=nvlink_peers,
                )

                # Add GPU to NUMA node's GPU list
                if numa_node in self.numa_nodes:
                    self.numa_nodes[numa_node].gpu_ids.append(i)

            pynvml.nvmlShutdown()
            logger.info(f"Detected {len(self.gpu_info)} GPUs across "
                       f"{len(set(g.numa_node for g in self.gpu_info.values()))} NUMA nodes")

        except ImportError:
            logger.warning("pynvml not available, GPU-NUMA mapping may be incomplete")
        except Exception as e:
            logger.warning(f"Failed to detect GPU-NUMA mapping: {e}")

    def _get_numa_node_for_pci(self, pci_bus: int) -> int:
        """Determine NUMA node for a given PCI bus."""
        try:
            # Search in /sys/bus/pci/devices/
            for root, dirs, files in os.walk('/sys/bus/pci/devices'):
                for d in dirs:
                    # Match bus pattern like 0000:00:, 0000:01:, etc.
                    if re.match(rf'^\d{{4}}:{pci_bus:02x}:', d):
                        numa_path = os.path.join(root, d, 'numa_node')
                        if os.path.exists(numa_path):
                            with open(numa_path, 'r') as f:
                                return int(f.read().strip())
        except Exception as e:
            logger.debug(f"Failed to determine NUMA node for PCI bus {pci_bus}: {e}")
        return 0

    def get_numa_node_for_gpu(self, gpu_id: int) -> int:
        """Get the NUMA node for a specific GPU."""
        if gpu_id in self.gpu_info:
            return self.gpu_info[gpu_id].numa_node
        return 0

    def get_gpus_in_numa(self, numa_node: int) -> List[int]:
        """Get all GPUs in a specific NUMA node."""
        if numa_node in self.numa_nodes:
            return self.numa_nodes[numa_node].gpu_ids
        return []

    def is_cross_numa(self, gpu1: int, gpu2: int) -> bool:
        """Check if communication between two GPUs crosses NUMA boundaries."""
        return self.get_numa_node_for_gpu(gpu1) != self.get_numa_node_for_gpu(gpu2)

    def get_nvlink_peers(self, gpu_id: int) -> List[int]:
        """Get list of GPUs connected via NVLink to the specified GPU."""
        if gpu_id in self.gpu_info:
            return self.gpu_info[gpu_id].nvlink_peers
        return []

    def get_summary(self) -> str:
        """Get a human-readable summary of the topology."""
        lines = [
            "=" * 60,
            "NUMA Topology Summary",
            "=" * 60,
            f"NUMA Nodes: {len(self.numa_nodes)}",
            f"GPUs: {len(self.gpu_info)}",
        ]

        for node_id, node_info in self.numa_nodes.items():
            mem_gb = node_info.total_memory / (1024**3)
            lines.append(f"\nNUMA Node {node_id}:")
            lines.append(f"  CPUs: {len(node_info.cpu_cores)} cores")
            lines.append(f"  GPUs: {node_info.gpu_ids}")
            lines.append(f"  Memory: {mem_gb:.1f} GB")

        lines.append("\nGPU Details:")
        for gpu_id, gpu_info in self.gpu_info.items():
            lines.append(f"  GPU {gpu_id}: NUMA {gpu_info.numa_node}, "
                        f"PCI {gpu_info.pci_bus_id}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def print_topology(self) -> None:
        """Print the topology information to stdout."""
        print(self.get_summary())


def detect_numa_topology() -> NumaTopology:
    """
    Convenience function to detect and return NUMA topology.

    Returns:
        NumaTopology: The detected NUMA topology

    Example:
        >>> topology = detect_numa_topology()
        >>> topology.print_topology()
    """
    return NumaTopology.detect()
