# 全片共享 SCS 模型（6,000 基因，流式稀疏输入）

这是在独立内存优化版上增加的新训练流程，不修改 `SCS/src`、旧整片队列及旧 checkpoint。
网络仍为原 SCS 的 8 层 Transformer、50 个输入点（中心 + 49 邻居）、16 类方向和前景预测。
默认特征数按本次要求设为 **6,000**，可用 `--n-genes` 调整；这不是已经证明优于 2,000 的准确性结论。

## 方法与内存

1. 先留出约 10% 的完整非空分块作为验证区域。每个训练/验证核心区再内缩 `halo + 10 × bin_size`（当前 90 个原始坐标单位），排除重复 halo 样本和跨集合邻域。验证块不参与特征选择或梯度更新；最终也使用同一模型推理。
2. 从其余各区域的内部 RNA bin 等额抽样（默认每块最多 512 个），以稀疏矩阵拟合 Seurat-v3 HVG（span=1）。固定 6,000 个基因及列顺序。保留原 SCS 的原始合并计数，不做各块独立的归一化或 HVG 选择。如果抽样中符合条件的基因不足 6,000，明确报错，而不静默减少特征。
3. 从既有无损空间索引重建特征，而不是拼接旧 `x_train`。每块只保存一次 CSR 表达矩阵、邻居索引以及标签；当前批次才转换为 float32 的 `batch × 50 × 6000`。不创建全片或整块的展开表达张量。
4. 所有训练区域更新**同一个模型、同一个优化器**。每轮随机化区域顺序和块内样本顺序，默认每块最多 4,096 个核内正样本，并选最多相同数量的背景负样本；没有正样本的区域仍可提供背景负样本。每轮重新采样，验证采样固定。
5. 训练后统一加载一个最佳 checkpoint，分块进行全量推理，再复用原先的局部后处理与 halo 合并。

核标签仍来自现有染色 watershed，是伪标签，不是真实人工细胞边界。模型质量需要额外的形态学/表达一致性评估。
采样器每轮访问所有有标签的训练区域，但**不保证每轮用完所有核内点**；每块上限是显式训练预算，不能把 100 轮直接等同于原来每块各 100 轮。
每个 batch 暂时来自一个区域以保持磁盘局部性，区域间不重置权重；并非每块独立训练。

## 背景、覆盖率与一致性保护

- 仅当核心区原始 RNA 记录为零时跳过。少核、无核、低表达不等于空白，不被静默丢弃。
- 背景负样本沿用 `stain <= 10` 且距核中心 `> 30` 的规则。RNA 阳性但没有核的块也可参与共享模型推理。
- 邻域沿用原实现的十层方环顺序，不改成欧氏 KNN。表达不足 50 个邻居的 bin 仍不能由原模型支持，单独记录 `insufficient_neighbor_bins`；推理覆盖所有支持的中心，而非仅训练样本。
- 模型、特征、空间划分、分块产物都带 schema 指纹。更换基因顺序/数量需要新运行，不能复用旧 checkpoint。
- 新版预测按空间中心顺序写出，每个支持中心一次；后处理按显式坐标读取，不依赖旧版“训练点在前”的行顺序。
- 检查点保存每轮的模型、AdamW 优化器状态和已完成轮数。`--resume` 从最近完整轮继续；中断中的一轮重新跑。GPU 随机运算不承诺跨进程逐位复现。
- 方向准确率只统计正样本，前景分类另外报告 accuracy/AUC；验证指标来自伪标签，不能当作细胞分割的真实准确率。
- 全片导出要求每块状态完整且模型指纹一致。小规模 `--tiles` 工程测试标记为 `partial_scope`，禁止伪装为全片导出。

## 在计算节点运行

先按项目资源规则复用已有 Slurm 作业，在计算节点激活 `.venv-scs`；长任务在 tmux 内运行。以下命令不申请资源，也不会自动停止旧队列。
工作目录为 `/home/glancerz/labwork/codex/TLS`，新运行目录应与旧目录分开。

```bash
source /home/glancerz/labwork/codex/TLS/.venv-scs/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export MPLBACKEND=Agg

# 仅首次初始化：统一特征与固定空间验证划分。
CUDA_VISIBLE_DEVICES= python -B -m optimizations.scs_streaming.shared_prepare init \
  --source runs/ST19_whole_scs --output runs/ST19_shared_6000 \
  --n-genes 6000 --n-neighbors 50

# 可以逐块续跑：完成标记只有在全部文件写完后出现。
CUDA_VISIBLE_DEVICES= python -B -m optimizations.scs_streaming.shared_prepare prepare \
  --output runs/ST19_shared_6000

python -B -m optimizations.scs_streaming.shared_train train \
  --root runs/ST19_shared_6000 --epochs 100 --batch-size 10 \
  --per-class-cap 4096 --gpu-memory-mib 16384

# 如果训练中断，原参数不变，额外加 --resume。
python -B -m optimizations.scs_streaming.shared_train predict \
  --root runs/ST19_shared_6000 --inference-batch-size 32 --gpu-memory-mib 16384

CUDA_VISIBLE_DEVICES= python -B -m optimizations.scs_streaming.shared_train postprocess \
  --root runs/ST19_shared_6000
CUDA_VISIBLE_DEVICES= python -B -m optimizations.scs_streaming.shared_train merge \
  --root runs/ST19_shared_6000
```

默认 16 GiB 是 TensorFlow allocator 上限，不是对任意 batch/邻域设置的显存保证。
6,000 基因相对 2,000 基因主要扩大输入和第一个投影层，不会把后续 64 维注意力层也扩大三倍。
共享模型改善区域泛化是待验证假设，不能仅凭实现方式保证效果更好或训练更快。

## 检查与真实数据工程测试

```bash
CUDA_VISIBLE_DEVICES= python -B -m unittest optimizations.scs_streaming.test_shared -v
```

测试覆盖：稀疏批次与展开数组完全一致；计数大于 255 不溢出；原方环邻域顺序；空/稀疏块；核心区边界；每轮区域采样；固定验证采样；不兼容基因顺序拒绝加载。

`scripts/scs_shared_smoke.sh` 使用 ST19 的 `x10_y11`、`x12_y11` 训练，`x11_y11` 验证，真实 6,000 基因，先 1 轮再恢复到 2 轮。
每块每类只采 32 个点，以验证工程链路；这不是最终模型，不用于判断分割质量。
随后两个训练块用同一 checkpoint 做完整推理，并对第一个块做原 SCS 后处理。
日志在 `optimizations/scs_streaming/benchmarks/shared_6000_smoke.log`，产物在 `runs/ST19_shared_6000_smoke`。

### 2026-09-05 已完成的验证

- 新共享流程 10 项测试及原流式优化 8 项回归测试，共 18 项通过。
- 真实 ST19 的 6,000 基因特征与原空间索引重新聚合后的计数逐项一致。
- 训练 1 轮后在新进程恢复到第 2 轮，优化器累计 16 次更新，与两轮各 8 个 batch 相符，没有重新初始化为逐块模型。
- 两块分别输出 192,528 和 192,518 个预测中心，共 385,046 个；坐标、行数、有限数值、概率范围和同一 checkpoint 指纹均通过检查。
- `x10_y11` 完成原 SCS 后处理，输出 uint32 标签，形状与分块 + halo 一致。
- 两次短训练进程峰值 RSS 约 1.50 / 1.21 GiB。这是每块每类最多 32 个采样点的工程测试，不是完整训练、推理和预处理全过程的内存上界。

机器可读检查结果：`runs/ST19_shared_6000_smoke/engineering_validation.json`。
旧队列保持原样；这次没有启动正式全片共享训练，也没有评估新方案的生物学分割质量。
