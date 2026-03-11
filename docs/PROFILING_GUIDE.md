# NSys / NCU 性能采集指南

本指南介绍如何使用 NVIDIA Nsight Systems (NSys) 和 Nsight Compute (NCU) 对 NCCL Ring AllReduce 和 NUMA AllReduce 进行性能采集和对比分析。

---

## 目录

1. [工具简介](#工具简介)
2. [前置准备](#前置准备)
3. [Nsight Systems (NSys) 采集](#nsight-systems-nsys-采集)
4. [Nsight Compute (NCU) 采集](#nsight-compute-ncu-采集)
5. [结果分析](#结果分析)
6. [快速参考命令](#快速参考命令)

---

## 工具简介

| 工具 | 用途 | 分析粒度 |
|------|------|---------|
| **Nsight Systems (NSys)** | 系统级性能分析 | CPU/GPU 交互、Kernel 调度、PCIe 传输 |
| **Nsight Compute (NCU)** | Kernel 级性能分析 | 单个 CUDA Kernel 的指令吞吐量、寄存器使用、共享内存等 |

---

## 前置准备

### 1. 确认工具安装

```bash
# 检查 NSys
nsys --version

# 检查 NCU
ncu --version
```

如果未安装，从 NVIDIA 官网下载并安装 CUDA Toolkit（包含 NSys 和 NCU）。

### 2. 编译 vLLM

```bash
cd /path/to/vllm
python setup.py build_ext --inplace
```

### 3. 设置环境变量

```bash
cd /path/to/numa-allreduce
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

---

## Nsight Systems (NSys) 采集

### 基本采集命令

```bash
cd /path/to/numa-allreduce

# 基础采集（推荐）
nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
    -o numa_allreduce_tp4 \
    python examples/profile_nsys_ncu.py --world_size 4 --mode nsys --size_mb 256
```

### 常用参数说明

| 参数 | 说明 |
|------|------|
| `-w true` | 等待程序结束后再退出 |
| `-t cuda,nvtx,osrt` | 追踪 CUDA、NVTX、OS runtime |
| `--cudabacktrace=true` | 启用 CUDA backtrace |
| `-o filename` | 输出文件名（无需扩展名） |
| `-s cpu` | 采样模式：cpu 或 none |
| `--sample-period=1000` | 采样周期（ns） |

### 不同配置的采集

```bash
# TP2 采集
nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
    -o numa_allreduce_tp2 \
    python examples/profile_nsys_ncu.py --world_size 2 --mode nsys --size_mb 256

# TP4 采集
nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
    -o numa_allreduce_tp4 \
    python examples/profile_nsys_ncu.py --world_size 4 --mode nsys --size_mb 256

# TP8 采集（跨 NUMA）
nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
    -o numa_allreduce_tp8 \
    python examples/profile_nsys_ncu.py --world_size 8 --mode nsys --size_mb 256

# 不同数据量
for size in 64 256 1024; do
    nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
        -o numa_allreduce_tp4_${size}mb \
        python examples/profile_nsys_ncu.py --world_size 4 --mode nsys --size_mb $size
done
```

### 查看结果

```bash
# 使用 GUI 查看
nsys-ui numa_allreduce_tp4.qdrep

# 或者导出为文本
nsys stats numa_allreduce_tp4.qdrep
```

### NVTX Markers

脚本中已包含 NVTX markers，在 NSys 中可以看到：
- `NCCL_AllReduce`: NCCL 执行区间
- `NCCL_Iter_*`: 每次 NCCL 迭代
- `NUMA_AllReduce`: NUMA AllReduce 执行区间
- `NUMA_Iter_*`: 每次 NUMA 迭代

---

## Nsight Compute (NCU) 采集

### 基本采集命令

```bash
cd /path/to/numa-allreduce

# 采集所有 Kernel（完整信息）
ncu --set full -o ncu_profile_tp4 \
    python examples/profile_nsys_ncu.py --world_size 4 --mode ncu --size_mb 256
```

### 常用参数说明

| 参数 | 说明 |
|------|------|
| `--set full` | 采集完整指标集 |
| `--set default` | 采集默认指标集（更快） |
| `--set detailed` | 采集详细指标集 |
| `-o filename` | 输出文件名 |
| `-k regex` | 只采集匹配正则的 Kernel |
| `--profile-from-start off` | 不从开头开始采集 |

### 只采集特定 Kernel

```bash
# 只采集 NCCL 相关 Kernel
ncu --set full -k "nccl" -o ncu_nccl_only \
    python examples/profile_nsys_ncu.py --world_size 4 --mode ncu --size_mb 256

# 只采集自定义 Kernel
ncu --set full -k "custom|allreduce" -o ncu_custom_only \
    python examples/profile_nsys_ncu.py --world_size 4 --mode ncu --size_mb 256
```

### 不同配置的采集

```bash
# TP2
ncu --set full -o ncu_tp2 \
    python examples/profile_nsys_ncu.py --world_size 2 --mode ncu --size_mb 256

# TP4
ncu --set full -o ncu_tp4 \
    python examples/profile_nsys_ncu.py --world_size 4 --mode ncu --size_mb 256

# TP8
ncu --set full -o ncu_tp8 \
    python examples/profile_nsys_ncu.py --world_size 8 --mode ncu --size_mb 256
```

### 查看结果

```bash
# 使用 GUI 查看
ncu-ui

# 或者导出为报告
ncu --import ncu_tp4.ncu-rep --csv
```

---

## 结果分析

### NSys 分析要点

| 指标 | 说明 | 关注内容 |
|------|------|---------|
| **Kernel 执行时间** | 每个 Kernel 的耗时 | NCCL vs NUMA 的总耗时对比 |
| **GPU 利用率** | GPU 忙碌时间比例 | 是否有 GPU 空闲 |
| **PCIe 传输** | PCIe 带宽使用 | NUMA 架构下的跨 NUMA 传输 |
| **NVTX Ranges** | 标记的执行区间 | 精确对比两种实现 |

### NCU 分析要点

| 指标 | 说明 |
|------|------|
| **Memory Throughput** | 内存吞吐量 |
| **Compute Throughput** | 计算吞吐量 |
| **SM Utilization** | SM 利用率 |
| **Warp Occupancy** | Warp 占用率 |
| **Shared Memory Usage** | 共享内存使用 |
| **Register Usage** | 寄存器使用 |

### 对比表格模板

| 配置 | Size | NCCL (ms) | NUMA (ms) | Speedup |
|------|------|-----------|-----------|---------|
| TP2 | 256 MB | | | |
| TP4 | 256 MB | | | |
| TP8 | 256 MB | | | |

---

## 快速参考命令

### 一键对比采集（TP2/TP4/TP8）

```bash
cd /path/to/numa-allreduce

# 创建输出目录
mkdir -p profiles

# TP2
nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
    -o profiles/numa_allreduce_tp2 \
    python examples/profile_nsys_ncu.py --world_size 2 --mode nsys --size_mb 256

# TP4
nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
    -o profiles/numa_allreduce_tp4 \
    python examples/profile_nsys_ncu.py --world_size 4 --mode nsys --size_mb 256

# TP8
nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
    -o profiles/numa_allreduce_tp8 \
    python examples/profile_nsys_ncu.py --world_size 8 --mode nsys --size_mb 256

echo "All profiles saved to ./profiles/"
```

### 快速基准测试（不使用 profiler）

```bash
# 运行基准测试
python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark --size_mb 256

# 多种数据量
for size in 64 256 1024; do
    echo "=== Size: ${size} MB ==="
    python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark --size_mb $size
done
```

### 不同数据类型测试

```bash
# FP32
python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark --dtype float32

# FP16
python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark --dtype float16

# BF16
python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark --dtype bfloat16
```

---

## 常见问题

### Q: NSys 报错 "CUDA backward is not available"

A: 确保使用 `--cudabacktrace=true` 参数。

### Q: NCU 采集速度太慢

A: 使用 `--set default` 而不是 `--set full`，或者用 `-k` 过滤只采集特定 Kernel。

### Q: 如何只采集 Rank 0 的数据？

A: 在脚本中添加条件判断，只在 Rank 0 时进行采集，或者使用 NSys/NCU 的过滤功能。

### Q: 采集文件太大怎么办？

A: 减少 `--iterations` 参数，或者减小 `--size_mb`。

---

## 参考资料

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiler User Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
