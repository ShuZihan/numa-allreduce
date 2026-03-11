# NUMA-Aware AllReduce

一个针对 PCIe-only 多 GPU 系统优化的自定义 AllReduce 算子，具备 NUMA 亲和性。

---

## 🚀 快速开始

```bash
# 1. 进入目录
cd numa_allreduce
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 2. 运行演示（4卡）
python examples/numa_allreduce_demo.py --world_size 4

# 3. 运行单元测试
pytest tests/test_numa_all_reduce.py -v
```

---

## 📁 目录结构

```
numa_allreduce/
├── README.md                    # 本文件（快速开始）
├── src/                         # 源代码
│   ├── __init__.py
│   ├── numa_all_reduce.cu       # CUDA 内核
│   ├── numa_all_reduce.py       # Python 封装
│   └── numa_utils.py            # NUMA 拓扑检测
├── tests/                       # 单元测试
│   └── test_numa_all_reduce.py  # pytest 测试套件
├── examples/                    # 示例
│   └── numa_allreduce_demo.py   # 交互式演示
└── docs/                        # 文档
    ├── UNIT_TEST_AND_BENCHMARK_GUIDE.md  # ⭐ 详细测试指南
    ├── NUMA_ALLREDUCE_README.md           # 设计文档
    └── BF16_SUPPORT.md                     # BF16 支持说明
```

---

## 📖 文档索引

| 文档 | 内容 | 何时阅读 |
|------|------|----------|
| **[docs/UNIT_TEST_AND_BENCHMARK_GUIDE.md](docs/UNIT_TEST_AND_BENCHMARK_GUIDE.md)** | ⭐ **推荐先读** - 完整的测试和性能比对流程 | 第一次使用 |
| [docs/NUMA_ALLREDUCE_README.md](docs/NUMA_ALLREDUCE_README.md) | 设计文档、算法说明、服务器拓扑分析 | 想了解实现细节 |
| [docs/BF16_SUPPORT.md](docs/BF16_SUPPORT.md) | BF16 数据类型支持的专项说明 | 使用 BF16 时 |

---

## 💡 使用示例

### 基本用法

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    from numa_all_reduce import NumaAwareAllReduce

    # 初始化
    na_ar = NumaAwareAllReduce(
        group=dist.group.WORLD,
        device=rank,
    )

    # 创建张量
    tensor = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')

    # 使用 NUMA-aware AllReduce
    if not na_ar.disabled:
        result = na_ar.custom_all_reduce(tensor)

    dist.destroy_process_group()

# 运行
mp.spawn(run_worker, args=(4,), nprocs=4, join=True)
```

### 运行演示

```bash
# TP2（2卡）
python examples/numa_allreduce_demo.py --world_size 2

# TP4（4卡）
python examples/numa_allreduce_demo.py --world_size 4

# TP8（8卡，跨 NUMA）
python examples/numa_allreduce_demo.py --world_size 8

# 只跑性能测试，跳过正确性
python examples/numa_allreduce_demo.py --world_size 4 --no-correctness
```

### 运行测试

```bash
# 所有测试
pytest tests/test_numa_all_reduce.py -v -s

# 只跑正确性测试
pytest tests/test_numa_all_reduce.py::TestNumaAllReduceCorrectness -v

# 只跑性能测试
pytest tests/test_numa_all_reduce.py::TestNumaAllReducePerformance -v

# 只跑 BF16 测试
pytest tests/test_numa_all_reduce.py::TestNumaAllReduceBf16 -v
```

---

## 🔧 支持的数据类型

| 类型 | 支持 |
|------|------|
| `torch.float32` | ✅ |
| `torch.float16` | ✅ |
| `torch.bfloat16` | ✅ |

---

## 📊 算法概述

采用三阶段分层 AllReduce 算法：

1. **Stage 1: Intra-NUMA Reduce-Scatter** - NUMA 节点内局部归约
2. **Stage 2: Inter-NUMA AllReduce** - NUMA 节点间交换
3. **Stage 3: Intra-NUMA AllGather** - NUMA 节点内广播

**优势**：跨 NUMA 流量从 100% 降到约 25%。

---

## 📚 更多信息

详细信息请参阅 [docs/](docs/) 目录下的文档。
