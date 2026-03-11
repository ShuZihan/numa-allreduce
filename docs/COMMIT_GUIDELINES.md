# Git Commit 规范

本文档定义了项目的 Git Commit Message 规范，旨在提高代码历史的可读性和可维护性。

---

## Commit Message 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 格式说明

1. **type**: commit 的类型（必填）
2. **scope**: commit 影响的范围（可选）
3. **subject**: commit 的简短描述（必填，不超过 50 字符）
4. **body**: commit 的详细描述（可选）
5. **footer**: 关联 Issue 或 Breaking Changes（可选）

---

## Type 类型

| 类型 | 说明 |
|------|------|
| `feat` | 新功能（feature） |
| `fix` | 修复 bug |
| `docs` | 文档更新 |
| `style` | 代码格式调整（不影响代码运行） |
| `refactor` | 重构（既不是新增功能，也不是修复 bug） |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `chore` | 构建过程或辅助工具的变动 |
| `revert` | 回滚之前的 commit |

---

## 示例

### 示例 1: 新功能

```
feat(allreduce): add NUMA-aware three-stage algorithm

- Add Stage 1: Intra-NUMA Reduce-Scatter
- Add Stage 2: Inter-NUMA AllReduce
- Add Stage 3: Intra-NUMA AllGather
- Add topology detection and GPU-NUMA mapping
```

### 示例 2: 修复 bug

```
fix(bf16): correct type casting for Ampere GPUs

Fix incorrect __bfloat162float usage on CUDA arch >= 800

Closes: #123
```

### 示例 3: 文档更新

```
docs(readme): reorganize folder structure and update paths

- Rename folder from numa_allreduce_package to numa_allreduce
- Update all path references in documentation
- Add directory structure diagram
```

### 示例 4: 重构

```
refactor(topology): simplify NUMA node detection logic

- Remove redundant nvidia-smi calls
- Cache topology results
- Improve error handling
```

### 示例 5: 性能优化

```
perf(allreduce): optimize P2P memory access patterns

- Use 128-bit vectorized loads/stores
- Reduce cross-NUMA traffic by 75%
- Benchmark shows 30-45% latency reduction for TP8
```

---

## Subject 规范

1. **使用动词开头**：使用现在时，如 `add`、`fix`、`update`，而不是 `added`、`fixed`
2. **首字母小写**：除非是专有名词
3. **结尾不加句号**
4. **不超过 50 字符**：保持简洁

**✅ 好的示例：**
```
feat(allreduce): add NUMA-aware algorithm
fix(bf16): correct type casting for Ampere
docs(readme): update getting started guide
```

**❌ 不好的示例：**
```
Added a new feature for NUMA.
Fixed some bugs in BF16 support.
Updating the README.md file with new information.
```

---

## Body 规范

1. **详细描述**：解释「为什么」做这个变更，而不是「怎么做」
2. **分段**：使用空行分隔不同的要点
3. **使用列表**：多个变更点使用 `-` 或 `*` 列表
4. **每行不超过 72 字符**：方便在终端查看

---

## Footer 规范

### 关联 Issue

```
Closes: #123
Fixes: #456
See also: #789
```

### Breaking Changes

```
BREAKING CHANGE: The `custom_all_reduce` API now requires a device parameter

Migration guide:
- Old: `na_ar.custom_all_reduce(tensor)`
- New: `na_ar.custom_all_reduce(tensor, device=rank)`
```

---

## 常见场景的 Commit Message

### 文件夹重命名

```
chore: rename folder from numa_allreduce_package to numa_allreduce

Update all path references in documentation and code.
```

### 合并文档

```
docs: merge topology analysis into main design doc

- Move topology_analysis.md content to NUMA_ALLREDUCE_README.md
- Add as Appendix section
- Remove redundant topology_analysis.md
```

### 添加测试

```
test: add BF16 correctness tests for TP8

- Test various tensor shapes (1D, 2D, 3D, 4D)
- Test with world_size=2, 4, 8
- Add performance benchmark for BF16
```

---

## 实用工具

### 使用 Commitizen（可选）

可以使用 Commitizen 工具来辅助生成符合规范的 commit message：

```bash
# 安装
npm install -g commitizen

# 使用
git cz
```

### 检查 Commit Message

可以使用 `commitlint` 来自动检查 commit message 是否符合规范。

---

## 参考

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Angular Git Commit Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format)
