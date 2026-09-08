# Cellist 与 full-gene GenePT 4× 的受控比较

执行目录：`runs/Cellist_full_genept_v1`。主结果以各区域的
`comparison_completed.json` 为准；正在运行的区域不纳入完成结论。

## 范围和资源

按用户确认，排除 ST19。包括 Stereo-seq 小鼠脑整片、Seq-Scope 肝脏
2104–2107、seqFISH+ NIH/3T3 实验 1 的 FOV 0 和 1，共七个评估单元，
属于三个数据来源；不是七个独立生物学重复，也不覆盖论文全部七个 seqFISH 视野。

- GPU allocation `20445437`，`rg31902`，2×H100、32 CPU、230 GB、4 小时。
  GPU 0 继续小鼠脑训练，GPU 1 依次训练两个 seqFISH 视野。
- 独立 CPU allocation `20445592`，`rc31706`，32 CPU、230 GB、4 小时，无 GPU。
  所有 Cellist、输入准备、后处理及比较均在此作业执行，不与 GPU 节点共享内存。
- `sbatch` 只运行 sleep 以保持 allocation。实际任务使用计算节点 tmux；
  各阶段使用独立 socket，避免一个 Slurm step 退出时清理其他阶段的服务器。
- 环境分别为 `.venv-scs-torch`、`scripts/cellist_env.sh` 和 `.venv-scs`。

## 方法和输入

GenePT 使用官方 `GenePT_gene_embedding_ada_text.pickle`，SHA256
`fd297510ddd3040744033fde0b0f2cf15a40ac8b2fd2fb02f10667295e55c862`。
所有能映射的基因均参与，无 2,000/HVG 截断。映射为大小写不敏感的精确符号匹配，
不是经过验证的人鼠同源映射。原始向量 1,536 维、冻结，计数加权求和除以
非零映射基因数；重复符号先合并计数。不能映射的基因不进入向量，但仍用于
空间邻域支持及共同的 RNA 评价分母。每区的 `prepared.json` 记录映射覆盖率。

4× 模型宽 256、32 层、4 头、中心加 49 邻居，100 epochs，Muon + AdamW。
验证联合 BCE/方向交叉熵最低的 checkpoint 用于推理，人工 IoU 不参与选模。
训练、验证来自同片，不能解释为跨切片泛化；16 类方向准确率不是分割准确率。
肝脏复用已经完成的相同全基因训练及推理，后处理重新执行；小鼠脑复制 checkpoint
后继续训练，不修改原始产物；seqFISH 重建共同网格后从头训练。

Cellist 使用官方 1.1.1 数值实现，源码提交
`7f6781786e8745fbc3569e1fc1f644c3e705be5e`。输入为所有原始测得基因，保留官方内部
每块 1,500 HVG 选择。已有工程适配包括可选依赖导入、线程数可配置及限制绘图尺寸。
本轮没有为了匹配 GenePT 而删除 Cellist 的特征选择。

两方法使用相同 RNA 范围、核图谱、配准、物理单位与外层分块。小鼠脑按原有
1,200 原始像素块计算，边界不做跨块合并；肝脏完整区域；seqFISH 使用共同
5×5 原始像素网格（0.515 µm/grid）。共享核为图像 watershed；Cellist 原文脑数据
使用 Cellpose。因此这是共享核条件下的受控比较，不是原生流程的精确论文复现。

## 已核实的参数

来源为作者 Zenodo 评估代码 Fig2/Fig3 的 Config。Cellist 均为 alpha=0.9、
sigma=1、beta=5、HVG、two_step=False、cyto=False。

| 区域 | Cellist max_dist | noise_prop | neigh_dist (µm) | SCS 后处理 r_estimate |
|---|---:|---:|---:|---:|
| 小鼠脑 | 15 | 0.45 | 2.5 | 15 原始像素 |
| 肝脏 | 25 | 0.30 | 3.0 | 25 原始像素 |
| seqFISH | 10 | 0 | 2.5 | 40 网格单位，即 200 原始像素 |

参数继承作者配置，不按最终分数挑选。本轮主输出为 `cellist_paper` 和
`genept_paperpost_scale4`。早期默认参数试跑保存在 `cellist`、
`evaluation_default_pilot`、`comparison_default_pilot.json`，不混入主结果。

单核脑边缘块已用 Python faulthandler 确认：官方 Seurat-v3 loess 会发生 native
segmentation fault。适配器只在少于 3 个核时使用所有核内表达基因，记录于
`hvg_compatibility.json`；其他 HVG 拟合保持官方算法并串行调用。每个外层块用独立
子进程，异常不会静默跳过。少于官方要求的 20 个观测核内 spot 的核不参与 Cellist；
没有合格核的块记录为未分配，RNA 仍保留在总体分母。兼容处理的块需在汇总中单列。

## 评价定义及限制

| 指标 | 方向 | 本轮定义 |
|---|---|---|
| seqFISH transcript IoU | 越高越好 | 按原始分子行的人工 cell ID 与预测归属做双向最大交叠的 mutual-best 匹配；同时报告作者 matched-only 均值/中位数、全部人工细胞漏检计零均值、IoU≥0.5 召回。不是稠密多边形 mask IoU。 |
| 随机相关性 | 越高越好 | 按作者代码随机切分 **spot 坐标**，每个 spot 的全部计数一起分配；计算聚合表达 Pearson。10 次共享种子的重复。并非按分子做二项拆分。 |
| 随机方向相关性 | 越高越好 | 用穿过各预测细胞 spot 质心的随机直线分割表达。每次方向对两方法相同，10 次重复。 |
| 双向 cross-correlation | 同一配对内越高越好 | source 每个细胞匹配最大 spot 交叠 target，允许 source 多对一合并；比较交集与两侧各自独有区。两个方向分开报告，共用非空且有限的合格配对；另存 100 UMI 敏感性标记。 |
| 邻域纯度比值 | 越低越好 | 共同表达经半径 floor(neigh_dist/resolution) 的 1/d² 空间核增强，self=1、不归一化；每细胞各基因样本方差/均值的中位数，除以共同核中心 ±20 坐标单位窗口的对应统计量。双方细胞和窗口均需 >100 观测 spot。 |

肝脏评价共用 1,000 个核表达 HVG，其余共用 1,500 个，Seurat-v3 span=1。
小鼠脑 HVG 在每个共享外层块计算，而非作者整片核图谱；退化块的共同基因规则明确记录。
纯度匹配依据共同核的最大观测 spot 交叠，重复 cell pair 数单列。窗口为含边界的
KD-tree 方窗；肝脏边长 24 µm，脑边长 20 µm，seqFISH 20.6 µm。
这些共同基因、匹配及重复随机种子规则是公平比较适配，不宣称逐值复现作者脚本。

先报告全部输出细胞、分配 RNA 比例、RNA/基因数，再报告 >100 spot 的相关性。
未分配位置保留为零标签；原始计数与坐标逐块检查守恒。因序列图邻居不足而没有
GenePT 预测的中心不被从共同 RNA 分母删除。失败块不被伪装成完成块。

seqFISH 原始 MAT 文件已按选定人工细胞整理分子；人工标签不进入训练，但输入范围
已经条件于这些人工细胞。故该 IoU 不是完全独立、全图无标注过滤的验证，也不能
据此报告全图 precision/AP。相关性和纯度都依赖分割时用过的表达，不等于独立边界真值；
更大的细胞、较高计数可能抬高相关性，必须连同覆盖与细胞数一起解读。

## 运行与检查

在独立 CPU allocation 内、激活对应环境后：

```bash
bash benchmarks/cellist_full_genept/launch.sh cpu
bash benchmarks/cellist_full_genept/launch.sh evaluation
```

GPU 任务使用 `launch_brain.sh` 和 `launch.sh seqfish`。这些 launcher 保持父 Slurm
step 存活，实际长任务在 tmux 运行。现有完成标记可跳过已完成工作。

已在 CPU 节点通过 10 项检查：稀疏 Pearson 对照密集计算、IoU 漏检和合并处理、
无独有区的 cross-correlation、随机种子与计数守恒、Fano 样本方差、
重复 RNA 合并与大计数、缺失基因拒绝、seqFISH 坐标不错误乘以 3、无核但有 RNA 的边缘情况（指标及完整分块评价两项）。
另已实际重跑导致 segfault 的单核脑块并完成输出。最终结果需检查全部七个区域
完成标记、每块 RNA 总数、模型选模记录和兼容处理覆盖。

`audit_seqfish.py` 另外用独立 pandas 列联表复算两个 FOV 的逐细胞 IoU，
并将分子坐标重新计数核对共同 RNA 矩阵及分配 RNA 总量；全部通过。
同坐标可视化保存在运行目录 `seqfish_fov0_shared_support.png` 和 FOV 1 对应文件。
`report.py` 每分钟更新比较报告、CSV 与检查结果；`provenance/` 保留代码副本、
哈希、环境版本和肝脏只读复用的指纹核查结果。

## 来源

- [Cellist 原文 DOI](https://doi.org/10.1038/s41588-026-02610-1)
- [作者完整 PDF](https://wanglabtongji.github.io/resources/publications/2026_NatGenet_Cellist.pdf)，SHA256 `40dd5ca7d46ad8e1f8e1176350f31fb8cc007e008720eb27c831318e9c545953`
- [作者评估代码 Zenodo 18638251](https://doi.org/10.5281/zenodo.18638251)，已下载 `Cellist_Evaluation.tar.gz`，MD5 `49dda0fa190baf3d52ab2c5353d5dfa7` 与记录一致。
- 本地来源保存在 `sources/`，主分析代码为 `Fig2/Fig2.py`、`Fig3/Fig3_Seqscope.py`、`Fig3/Fig3_seqFISH.py`。
