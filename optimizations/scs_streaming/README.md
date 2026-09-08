# SCS 内存与数据输入优化（独立实现）

本目录保留 `SCS/src` 的原始模型与算法。经用户授权，`scripts/scs_whole_stage.py` 的后续逐块训练现已接入内存优化；已在运行的训练进程不被中断，已完成分块不重跑。
使用现有 `.venv-scs` 环境；模型结构来自本次优化开始时的 SCS 源码快照。

## 默认实现

`train.py --input-mode arrays` 保留原有 Keras 训练接口，解决内存重复占用：

- NPZ 按流解压至计算节点临时目录，再以原始整数类型映射。
- 仅选中的训练集和验证集转换为 float32，不保留整份 float32 原始训练数组。
- 训练期间完全不加载 `x_test`；训练与验证的数据内容、空间划分规则保持不变。
- 训练结束后，按小批次读取推理数据，并逐批写出预测，保留训练点在前、测试点在后的 SCS 行顺序。
- 训练仍默认 100 轮、batch size 10，使用原架构、损失函数、AdamW 参数及 checkpoint 选择规则。
- 默认显存上限 16 GiB。对更大分块需重新核算；这不是任意数据规模都适用的保证。

另保留 `tensor` 和 `stream` 两个实验模式，方便后续研究。它们进一步降低内存，但本次测试更慢，未选为默认。

## 真实数据基准

输入为 `runs/ST19_whole_scs/tiles/x12_y11/data`，包括 26,495 个训练/背景样本，训练子集每轮 2,322 个 batch。
每种优化方案仅测试 3 轮，属于工程基准，不是用于细胞分割的完整训练。

| 实现 | 训练进程峰值 RSS | 第 2、3 轮耗时 | 显存上限 |
|---|---:|---:|---:|
| 原版主任务 | 实测约 59.2 GiB | 独占时通常约 21 秒/轮 | 28,000 MiB |
| Python 按批读取 `stream` | 6.19 GiB | 98.17 / 100.77 秒 | 4 GiB |
| 单份整数 CPU 缓存 `tensor` | 15.74 GiB | 60.85 / 59.68 秒 | 4 GiB |
| 精简数组加载 `arrays`（默认） | 28.75 GiB | 27.69 / 27.88 秒 | 16 GiB |

优化测试与主任务共用同一张 H100。上述时间不是独占 GPU 的严格 A/B 对照，不能据此声称单块速度提升或全片并发吞吐翻倍。
可靠结论是：默认版本在同一真实分块上的峰值 RSS 比原任务低约 51%，并通过了三轮训练。
默认版本的 TensorFlow GPU allocator 峰值约 10,155 MiB；4 GiB 上限不足以容纳这个 Keras 数组输入模式，相关失败测试单独保留。

基准记录：

- `benchmarks/x12_y11_arrays16_three_epochs/completed.json`
- `benchmarks/x12_y11_arrays16_three_epochs/training_history.csv`
- `benchmarks/x12_y11_three_epochs/completed.json`（最初的纯流式版本）
- `benchmarks/x12_y11_tensor_three_epochs/completed.json`

## 正确性检查

`test_streaming.py` 检查只读输入、整数保留、坐标转换、严格的验证区域边界、每轮采样覆盖、空测试集、预测顺序和原模型训练步骤。

`validate_real.py` 读取已完成的 `x10_y11` 原模型 checkpoint，抽取 32 个训练点和 32 个测试点，对照原批次大小 10 与较大的推理批次。
CPU 对照中，方向类别和前景阈值判断全部一致，最大 logit 绝对差约 `8.94e-8`，概率输出无差异。
GPU 对照的同一批 64 个点中，logit 和概率输出的绝对差均为 0；记录见 `benchmarks/real_checkpoint_equivalence_gpu.json`。
这是实现一致性检查，不等于分割准确性评估。

```bash
# 在计算节点的现有环境内运行；CPU 检查不会占用 GPU。
CUDA_VISIBLE_DEVICES= python -B -m unittest optimizations.scs_streaming.test_streaming -v
```

## 使用

在计算节点的 tmux 会话中，激活现有环境，并从项目根目录运行：

```bash
source /home/glancerz/labwork/codex/TLS/.venv-scs/bin/activate
python -B -m optimizations.scs_streaming.train \
  --data-dir runs/ST19_whole_scs/tiles/x12_y11/data \
  --output-dir runs/ST19_whole_scs/tiles/x12_y11/optimized_training \
  --epochs 100 --input-mode arrays --gpu-memory-mib 16384
```

输出目录必须与原运行目录分开，已有训练历史的目录会被拒绝。原始输入和原 checkpoint 保留。
该命令完成训练与推理，输出 `results/spot_prediction_0:0:0:0.txt`；后续细胞归属/掩膜仍需 SCS 后处理。
临时解压缓存默认位于 `SLURM_TMPDIR`，正常退出时自动清理；训练中断不会损坏原 NPZ。

整片并行入口为 `scripts/scs_whole_parallel.py`，由计算节点中的 `scripts/scs_parallel_tmux.sh` 启动。
跨作业共享逐块文件锁，逐阶段复用检查点；各 allocation 使用独立资源账本，同时限制主机内存与 GPU 显存。
单卡 120 GiB 作业的主机内存预算为 103.2 GiB，训练进程显存至少 16 GiB，较大分块按数组大小上调；每卡最多 4 个训练进程，实际并发受内存估算限制。
旧控制器保留，其后续阶段也参加资源租约；切换前已经运行的旧阶段按保守内存估计计入预算。
完整逐块训练保持 100 轮、训练和推理 batch size 10、原验证划分与 checkpoint 选择规则。每次训练写入独立 attempt 目录，推理成功后才原子发布结果并继续原 SCS 后处理。
运行记录位于 `runs/ST19_whole_scs/runtime/<jobid>/`。不自动续申请 allocation；未完成的训练 attempt 保留，但目前没有跨 attempt 恢复 optimizer 状态的机制。

## 全片共享模型的边界

新增共享模型入口见 [SHARED_MODEL.md](SHARED_MODEL.md)。它使用统一的 6,000 基因表、稀疏表达矩阵与邻居索引、区域平衡采样和一套共享权重；默认旧单块训练入口不变。
共享模型是待验证的方法改动，工程测试不等于完成全片共享训练或证明分割质量改善。
`x10_y11` 和 `x12_y11` 各自选出的 2,000 个高变基因只有 336 个重合，不能直接拼接现有特征张量。
需要统一基因集合及列顺序，从不同组织区域分层采样，留出完整区域并排除重叠边缘泄漏，才能评估区域泛化。
合并所有样本后仍训练 100 轮并不会自动减少样本总更新次数；加速要来自代表性采样、合理训练预算或有验证依据的提前停止。
