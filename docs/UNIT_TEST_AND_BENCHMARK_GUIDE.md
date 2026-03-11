# NUMA AllReduce 单元测试和性能比对指南

> **第一阶段目标**：仅对 NUMA AllReduce 做单元测试，并与原生 `torch.distributed.all_reduce` 做性能比对。**暂不涉及 vLLM 端到端验证**。

---

## 目录

1. [前提条件检查](#前提条件检查)
2. [编译 CUDA 算子](#编译-cuda-算子)
3. [快速开始](#快速开始)
4. [单元测试](#单元测试)
5. [性能比对测试](#性能比对测试)
6. [NSys / NCU 性能采集](#nsys--ncu-性能采集)
7. [快速开始命令](#快速开始命令)

---

## 前提条件检查

在开始之前，请确保以下条件满足：

```bash
# 1. 检查 CUDA 可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# 2. 检查 GPU 架构（需要 Ampere+ 以支持 BF16）
python -c "import torch; print('GPU capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"

# 3. 检查 P2P 访问状态
nvidia-smi topo -p2p r

# 4. 检查 NUMA 拓扑
numactl --hardware

# 5. 检查当前目录
pwd  # 应该在 numa-allreduce/ 目录
ls -la src/numa_all_reduce.cu       # 确认文件存在
ls -la tests/test_numa_all_reduce.py  # 确认测试文件存在
```

### 预期输出示例

```
CUDA available: True
GPU count: 8
GPU capability: (8, 0)  # Ampere 架构
```

---

## 编译 CUDA 算子

NUMA AllReduce 的 CUDA 算子需要编译后才能使用。有两种方式：

### 方式一：在 vLLM 源码树中编译（推荐）

如果您是在 vLLM 源码树中工作，算子会随 vLLM 一起编译：

```bash
# 进入 vLLM 源码根目录
cd /path/to/vllm

# 编译 vLLM（包含 custom_all_reduce 算子）
pip install -e .

# 或者仅编译 C++ 扩展（更快）
python setup.py build_ext --inplace
```

vLLM 的 `setup.py` 会自动编译 `csrc/custom_all_reduce.cu` 并生成 `vllm._C` 扩展模块。

### 方式二：独立编译（仅用于单元测试）

如果您想在独立的 numa-allreduce 仓库中编译，可以创建一个简单的 `setup.py`：

```bash
cd /path/to/numa-allreduce

# 创建临时 setup.py
cat > setup.py << 'EOF'
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="numa_allreduce_ops",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "numa_allreduce_ops",
            ["src/numa_all_reduce.cu"],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
EOF

# 编译
python setup.py build_ext --inplace
```

### 验证编译成功

```bash
# 检查是否可以导入 custom ops
python -c "
try:
    import vllm._custom_ops as ops
    print('✅ vLLM custom ops imported successfully')
    print(f'   meta_size() available: {hasattr(ops, \"meta_size\")}')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    print('   Please compile vLLM first.')
"
```

### 编译依赖

- CUDA Toolkit >= 11.8
- PyTorch with CUDA support
- CMake >= 3.26
- Ninja build system

**注意**：对于 BF16 支持，需要：
- CUDA 架构 >= 8.0 (Ampere or later)
- 在编译时会自动检测并启用 BF16 支持

---

## 快速开始

### 步骤 1: 进入目录并添加到 PYTHONPATH

```bash
cd /path/to/numa-allreduce
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### 步骤 2: 验证导入

```bash
# 测试导入
python -c "import numa_all_reduce; import numa_utils; print('✅ All imports OK')"
```

---

## 单元测试

### 方式一：使用 pytest 运行测试

```bash
# 进入目录
cd /path/to/numa-allreduce
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 设置 pytest 输出更详细
export PYTEST_ADDOPTS="-v -s"

# 1. 运行所有 NUMA 相关测试
pytest tests/test_numa_all_reduce.py

# 2. 只运行正确性测试
pytest tests/test_numa_all_reduce.py::TestNumaAllReduceCorrectness

# 3. 只运行 BF16 测试
pytest tests/test_numa_all_reduce.py::TestNumaAllReduceBf16

# 4. 只运行性能测试
pytest tests/test_numa_all_reduce.py::TestNumaAllReducePerformance

# 5. 运行特定 TP 配置的测试
pytest tests/test_numa_all_reduce.py::TestNumaAllReduceCorrectness::test_correctness_against_nccl -k "world_size=4"

# 6. 运行特定数据类型的测试
pytest tests/test_numa_all_reduce.py::TestNumaAllReduceCorrectness::test_correctness_against_nccl -k "bfloat16"
```

### 方式二：使用示例脚本运行测试

```bash
# TP2 测试
python examples/numa_allreduce_demo.py --world_size 2

# TP4 测试
python examples/numa_allreduce_demo.py --world_size 4

# TP8 测试（需要 8 卡）
python examples/numa_allreduce_demo.py --world_size 8

# 只运行正确性测试，跳过性能测试
python examples/numa_allreduce_demo.py --world_size 4 --no-performance

# 只运行性能测试，跳过正确性测试
python examples/numa_allreduce_demo.py --world_size 4 --no-correctness
```

### 方式三：使用交互式 Python 测试

创建一个临时测试脚本 `test_numa_simple.py`：

```python
#!/usr/bin/env python3
# test_numa_simple.py
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_test(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    try:
        # 导入 NUMA AllReduce
        from numa_all_reduce import NumaAwareAllReduce

        # 初始化
        na_ar = NumaAwareAllReduce(
            group=dist.group.WORLD,
            device=rank,
        )

        if rank == 0:
            print(f"NUMA AllReduce disabled: {na_ar.disabled}")

        # 创建测试张量
        torch.manual_seed(42 + rank)
        tensor = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')

        # 使用 NCCL 计算期望值
        expected = tensor.clone()
        dist.all_reduce(expected)

        # 使用 NUMA AllReduce
        if not na_ar.disabled:
            result = na_ar.custom_all_reduce(tensor.clone())

            if result is not None:
                # 验证正确性
                max_diff = (result - expected).abs().max().item()
                if rank == 0:
                    print(f"Max difference: {max_diff}")
                    print(f"Correct: {max_diff < 1e-3}")

    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = min(4, torch.cuda.device_count())
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
```

运行：

```bash
python test_numa_simple.py
```

---

## 性能比对测试

### 测试 1: 使用示例脚本进行完整性能比对

```bash
# TP2 性能测试
python examples/numa_allreduce_demo.py --world_size 2 --no-correctness

# TP4 性能测试（推荐，覆盖您的服务器典型配置）
python examples/numa_allreduce_demo.py --world_size 4 --no-correctness

# TP8 性能测试（跨 NUMA 节点，最能体现 NUMA 感知优势）
python examples/numa_allreduce_demo.py --world_size 8 --no-correctness
```

### 测试 2: 使用 pytest 性能测试

```bash
pytest tests/test_numa_all_reduce.py::TestNumaAllReducePerformance -v -s
```

### 测试 3: 自定义性能基准测试

创建 `benchmark_numa.py`：

```python
#!/usr/bin/env python3
"""
NUMA AllReduce vs torch.distributed.all_reduce 性能基准测试
"""
import os
import sys
import time
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def benchmark_worker(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    try:
        from numa_all_reduce import NumaAwareAllReduce

        na_ar = NumaAwareAllReduce(
            group=dist.group.WORLD,
            device=rank,
        )

        if rank == 0:
            print(f"{'='*70}")
            print(f"NUMA AllReduce vs torch.distributed.all_reduce Benchmark")
            print(f"World Size: {world_size} GPUs")
            print(f"{'='*70}\n")

        # 测试配置
        sizes_mb = config.get('sizes_mb', [64, 256, 1024])
        iterations = config.get('iterations', 100)
        warmup = config.get('warmup', 20)
        dtype_name = config.get('dtype', 'float32')
        dtype = getattr(torch, dtype_name)

        results = []

        for size_mb in sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            elem_size = 4 if dtype_name == 'float32' else 2
            numel = size_bytes // elem_size

            # 创建测试张量
            tensor = torch.randn(numel, dtype=dtype, device='cuda')

            # ========== Benchmark NCCL ==========
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

            # ========== Benchmark NUMA AllReduce ==========
            numa_times = []

            if not na_ar.disabled:
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
            else:
                numa_avg = float('nan')
                numa_std = float('nan')
                speedup = 0.0

            results.append({
                'size_mb': size_mb,
                'nccl_avg': nccl_avg,
                'nccl_std': nccl_std,
                'numa_avg': numa_avg,
                'numa_std': numa_std,
                'speedup': speedup,
            })

            # 打印结果
            if rank == 0:
                print(f"Size: {size_mb} MB ({dtype_name})")
                print(f"  {'Implementation':<15} {'Avg (ms)':<12} {'Std (ms)':<12} {'Bandwidth (GB/s)':<18}")
                print(f"  {'-'*60}")

                nccl_bw = (size_bytes * 2 / 1e9) / (nccl_avg / 1000)
                print(f"  {'NCCL':<15} {nccl_avg:<12.3f} {nccl_std:<12.3f} {nccl_bw:<18.2f}")

                if numa_times:
                    numa_bw = (size_bytes * 2 / 1e9) / (numa_avg / 1000)
                    print(f"  {'NUMA-Aware':<15} {numa_avg:<12.3f} {numa_std:<12.3f} {numa_bw:<18.2f}")
                    print(f"  Speedup: {speedup:.2f}x")
                else:
                    print(f"  {'NUMA-Aware':<15} {'DISABLED':<12}")
                print()

        # 汇总表格
        if rank == 0:
            print(f"{'-'*70}")
            print("SUMMARY:")
            print(f"{'-'*70}")
            print(f"{'Size (MB)':<12} {'NCCL (ms)':<12} {'NUMA (ms)':<12} {'Speedup':<10}")
            print(f"{'-'*48}")
            for r in results:
                if not (r['speedup'] > 0):
                    continue
                print(f"{r['size_mb']:<12} {r['nccl_avg']:<12.3f} {r['numa_avg']:<12.3f} {r['speedup']:<10.2f}")
            print(f"{'-'*70}\n")

    finally:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='NUMA AllReduce Benchmark')
    parser.add_argument('--world_size', type=int, default=4,
                       help='Number of GPUs (default: 4)')
    parser.add_argument('--dtype', type=str, default='float32',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Data type (default: float32)')
    parser.add_argument('--sizes', type=str, default='64,256,1024',
                       help='Comma-separated sizes in MB (default: 64,256,1024)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations (default: 100)')
    parser.add_argument('--warmup', type=int, default=20,
                       help='Number of warmup iterations (default: 20)')

    args = parser.parse_args()

    if args.world_size > torch.cuda.device_count():
        print(f"Error: Need {args.world_size} GPUs, but only {torch.cuda.device_count()} available")
        sys.exit(1)

    config = {
        'dtype': args.dtype,
        'sizes_mb': [int(s) for s in args.sizes.split(',')],
        'iterations': args.iterations,
        'warmup': args.warmup,
    }

    mp.spawn(
        benchmark_worker,
        args=(args.world_size, config),
        nprocs=args.world_size,
        join=True,
    )

if __name__ == '__main__':
    main()
```

运行基准测试：

```bash
# FP32, TP4, 64/256/1024 MB
python benchmark_numa.py --world_size 4 --dtype float32

# FP16, TP4
python benchmark_numa.py --world_size 4 --dtype float16

# BF16, TP4
python benchmark_numa.py --world_size 4 --dtype bfloat16

# BF16, TP8, 更大的数据量
python benchmark_numa.py --world_size 8 --dtype bfloat16 --sizes 256,512,1024,2048

# 快速测试（较少迭代）
python benchmark_numa.py --world_size 4 --iterations 20 --warmup 5
```

---

## 快速开始命令

### TL;DR: 完整测试流程（复制粘贴即可）

```bash
# ========== 阶段 1: 编译 vLLM（包含 CUDA 算子） ==========
# 如果还没编译 vLLM，先进入 vLLM 源码目录编译
cd /path/to/vllm
pip install -e .
# 或者更快的方式，仅编译扩展
python setup.py build_ext --inplace

# ========== 阶段 2: 进入 numa-allreduce 目录并设置环境 ==========
cd /path/to/numa-allreduce
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "=== Checking Environment ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

# ========== 阶段 3: 验证 custom ops 编译成功 ==========
echo -e "\n=== Verifying Custom Ops ==="
python -c "
try:
    import vllm._custom_ops as ops
    print('✅ vLLM custom ops imported successfully')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    print('   Please compile vLLM first.')
    import sys
    sys.exit(1)
"

# ========== 阶段 4: 验证导入 ==========
echo -e "\n=== Verifying Imports ==="
python -c "import numa_all_reduce; import numa_utils; print('✅ All imports OK')"

# ========== 阶段 5: 运行单元测试 ==========
echo -e "\n=== Running Unit Tests ==="
pytest tests/test_numa_all_reduce.py::TestNumaAllReduceCorrectness -v -s

# ========== 阶段 6: 运行性能比对 (TP4) ==========
echo -e "\n=== Running Performance Benchmark (TP4) ==="
python examples/numa_allreduce_demo.py --world_size 4 --no-correctness

# ========== 阶段 7: 运行性能比对 (TP8) ==========
echo -e "\n=== Running Performance Benchmark (TP8) ==="
python examples/numa_allreduce_demo.py --world_size 8 --no-correctness
```

### 针对您的服务器的推荐测试命令

```bash
# 您的服务器配置：
# - 8x GPU, 2x NUMA nodes
# - NUMA 0: GPU0-3
# - NUMA 1: GPU4-7

# 1. 同 NUMA 内测试 (GPU0-1, TP2)
python examples/numa_allreduce_demo.py --world_size 2

# 2. 同 NUMA 内测试 (GPU0-3, TP4)
python examples/numa_allreduce_demo.py --world_size 4

# 3. 跨 NUMA 测试 (GPU0-7, TP8) - 最关键！
python examples/numa_allreduce_demo.py --world_size 8
```

---

## NSys / NCU 性能采集

如需使用 NVIDIA Nsight Systems (NSys) 和 Nsight Compute (NCU) 进行深度性能分析，
请参考专门的性能采集指南：

**[PROFILING_GUIDE.md](./PROFILING_GUIDE.md)** - 完整的 NSys/NCU 性能采集指南

### 快速开始

```bash
cd /path/to/numa-allreduce
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 1. 快速基准测试（不使用 profiler）
python examples/profile_nsys_ncu.py --world_size 4 --mode benchmark --size_mb 256

# 2. 使用 NSys 采集系统级性能
nsys profile -w true -t cuda,nvtx,osrt --cudabacktrace=true \
    -o numa_profile_tp4 \
    python examples/profile_nsys_ncu.py --world_size 4 --mode nsys --size_mb 256

# 查看 NSys 结果
nsys-ui numa_profile_tp4.qdrep

# 3. 使用 NCU 采集 Kernel 级性能
ncu --set full -o ncu_profile_tp4 \
    python examples/profile_nsys_ncu.py --world_size 4 --mode ncu --size_mb 256
```

### 性能采集脚本

提供了专门的性能采集脚本 `examples/profile_nsys_ncu.py`，支持三种模式：

| 模式 | 说明 |
|------|------|
| `benchmark` | 仅运行基准测试，输出性能对比 |
| `nsys` | 为 NSys 采集添加 NVTX markers |
| `ncu` | 为 NCU 采集运行 Kernel 迭代 |

详细说明请参考 [PROFILING_GUIDE.md](./PROFILING_GUIDE.md)。

---

## 常见问题排查

### 问题 0: custom ops 未编译

```
ImportError: No module named 'vllm._custom_ops'
```

**解决方案**：

```bash
# 进入 vLLM 源码目录
cd /path/to/vllm

# 编译 vLLM
pip install -e .

# 或者仅编译 C++ 扩展
python setup.py build_ext --inplace
```

验证：
```bash
python -c "import vllm._custom_ops as ops; print('OK')"
```

### 问题 1: 导入错误 "No module named 'numa_all_reduce'"

```bash
# 确保在正确的目录
cd /path/to/numa-allreduce

# 添加 src 目录到 PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 然后重试
python -c "import numa_all_reduce; print('OK')"
```

### 问题 2: P2P 访问失败

```bash
# 设置环境变量跳过 P2P 检查
export VLLM_SKIP_P2P_CHECK=1

# 然后重新运行测试
```

### 问题 3: NUMA AllReduce 显示为 disabled

可能原因：

- world_size = 1
- P2P 访问未启用
- GPU 拓扑不是 fully connected

检查方式：

```python
from numa_utils import NumaTopology
topo = NumaTopology.detect()
topo.print_topology()
```

---

## 测试结果记录模板

创建 `test_results.md` 记录您的测试结果：

```markdown
# NUMA AllReduce 测试结果

## 环境信息

- 日期：YYYY-MM-DD
- GPU: [例如：8x A100]
- CUDA 版本：[例如：12.1]
- PyTorch 版本：[例如：2.2.0]

## 导入测试

- [ ] `import numa_all_reduce` 成功
- [ ] `import numa_utils` 成功

## 单元测试

- [ ] TP2 正确性测试通过 (FP32)
- [ ] TP2 正确性测试通过 (FP16)
- [ ] TP2 正确性测试通过 (BF16)
- [ ] TP4 正确性测试通过 (FP32)
- [ ] TP4 正确性测试通过 (FP16)
- [ ] TP4 正确性测试通过 (BF16)
- [ ] TP8 正确性测试通过 (FP32)
- [ ] TP8 正确性测试通过 (FP16)
- [ ] TP8 正确性测试通过 (BF16)

## 性能测试结果

### TP4 性能 （同 NUMA)

| Size (MB) | NCCL (ms) | NUMA (ms) | Speedup |
| --------- | --------- | --------- | ------- |
| 64        |           |           |         |
| 256       |           |           |         |
| 1024      |           |           |         |

### TP8 性能 （跨 NUMA)

| Size (MB) | NCCL (ms) | NUMA (ms) | Speedup |
| --------- | --------- | --------- | ------- |
| 64        |           |           |         |
| 256       |           |           |         |
| 1024      |           |           |         |

## 备注

[在此记录任何观察到的问题或异常]
```
