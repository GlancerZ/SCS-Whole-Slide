"""Collect only completed scores; preserve partial status and canonical report data.

Run on the compute node. This does not render or publish a report, change models,
select checkpoints, or use test results for fitting. No uncompleted score is zero-filled.
"""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
root=ROOT/'runs/SCS_paper_benchmark_v1'
METHODS=['scs_reference','raw_scale4','genept_scale4']
LABELS={'scs_reference':'SCS 基线','raw_scale4':'Raw 4×','genept_scale4':'GenePT 4×',
        'direction_prior_only':'仅方向先验','raw_matched_scale4':'Raw 4×（匹配覆盖）'}
TILES=['2104','2105','2106','2107','seqfish_rep1_fov0','seqfish_rep1_fov1']


def read(path):
    return json.loads(path.read_text())


def collect():
    now=datetime.now(timezone.utc).isoformat()
    records,curve,comparison,quality=[],[],[],[]
    full_segments=0
    for tile in TILES:
        for method in METHODS:
            d=root/tile/method
            row=dict(tile=tile,method=LABELS[method],method_id=method,status='待运行',
                     direction_accuracy=None,foreground_bacc=None,selected_epoch=None,
                     test_foreground_n=None,test_background_n=None,training_seconds=None)
            if (d/'config.json').exists(): row['status']='训练中'
            if (d/'completed.json').exists():
                result=read(d/'completed.json');t=result['test']
                # Independently recompute direction accuracy from saved counts.
                cm=t['direction_confusion']
                correct=sum(cm[i][i] for i in range(16));denominator=sum(map(sum,cm))
                if abs(correct/denominator-t['direction_accuracy'])>1e-12: raise ValueError('metric mismatch')
                if denominator!=t['foreground_n']: raise ValueError('denominator mismatch')
                row.update(status='已完成',direction_accuracy=t['direction_accuracy'],
                           direction_correct=correct,foreground_bacc=t['foreground_balanced_accuracy'],
                           selected_epoch=result['selected_epoch'],test_foreground_n=t['foreground_n'],
                           test_background_n=t['background_n'],training_seconds=result['elapsed_seconds'])
            records.append(row)
            if tile=='2104' and (d/'history.csv').exists():
                with (d/'history.csv').open() as f:
                    for h in csv.DictReader(f):
                        curve.append(dict(epoch=int(h['epoch']),method=LABELS[method],
                                          validation_direction=float(h['val_direction_accuracy']),
                                          train_direction=float(h['train_direction_accuracy']),
                                          train_loss=float(h['train_loss']),
                                          validation_foreground_bacc=float(h['val_foreground_bacc']),
                                          epoch_seconds=float(h['epoch_seconds']),tile=tile))
        full=root/tile/'benchmark.json'
        file=full if full.exists() else root/tile/'partial_benchmark.json'
        if full.exists(): full_segments+=1
        if file.exists():
            e=read(file)
            for p in e['paired_rna_consistency']:
                comparison.append(dict(tile=tile,first=LABELS[p['first']],second=LABELS[p['second']],
                                       correlation_first=p['correlation_first_mean'],correlation_second=p['correlation_second_mean'],
                                       difference=p['mean_paired_difference_first_minus_second'],
                                       eligible_nuclei=p['eligible_nuclei'],unique_pairs=p['unique_matched_pairs']))
            for method,q in e['quality'].items(): quality.append(dict(tile=tile,method=LABELS[method],**q))
    coverage=[]
    for r in read(root/'integrity_audit.json'):
        coverage.append(dict(tile=r['tile'],dataset='肝脏 Seq-scope' if r['tile'].isdigit() else 'NIH3T3 seqFISH+',
                             umi_coverage=r['mapped_umi_fraction'],genes_matched=r['mapped_genes'],genes_total=r['total_genes'],
                             bins_empty_after_mapping=r['bins_empty_after_genept_mapping'],train_n=r['train'],
                             validation_n=r['validation'],test_n=r['test']))
    completed=sum(r['status']=='已完成' for r in records)
    summary=dict(generated_at=now,completed_training=completed,planned_training=len(records),
                 fully_evaluated_sections=full_segments,records=records,coverage=coverage,
                 paired_rna_comparisons=comparison,segmentation_quality=quality,
                 coverage_ablation_complete=(root/'2104/coverage_ablation.json').exists())
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (root/'summary.csv').open('w') as f:
        fields=sorted(set().union(*(r.keys() for r in records)))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    title='SCS 原文数据集 Benchmark：阶段评估'
    section2104=[r for r in records if r['tile']=='2104']
    scores='；'.join(f"{r['method']} {r['direction_accuracy']:.2%}" for r in section2104 if r['direction_accuracy'] is not None)
    body=f'已完成 {completed}/{len(records)} 次主要训练、{full_segments}/6 个区域的完整评估。肝脏 2104 的已完成测试方向准确率：{scores}。尚未完成的模型不参与成绩排序。\n\n现有证据不能证明 GenePT 新方法已优于 SCS。需要同时检查最终分割指标、人工轮廓 IoU 和基因覆盖对照，不能仅凭训练或方向准确率判断。'
    sources=[dict(id='results',label='本次基准：完成状态与测试、分割指标',path='runs/SCS_paper_benchmark_v1/summary.json',
                  query={'description':'从完成标记、固定测试集预测和配对分割指标汇总；不补填未完成值。',
                         'sql':'python benchmarks/scs_paper/collect.py','engine':'bash','language':'bash',
                         'tables_used':['runs/SCS_paper_benchmark_v1/summary.json'],
                         'metric_definitions':{'direction_accuracy':'正确方向的前景测试中心点 / 全部前景测试中心点；不含背景。',
                                               'foreground_bacc':'(前景召回率 + 背景特异度) / 2。',
                                               'correlation':'交集表达与各自独有区表达的 Pearson r；三个区域均>=100 UMI。'}}),
             dict(id='curve',label='肝脏 2104：逐 epoch 保存的训练记录',path='runs/SCS_paper_benchmark_v1/2104',
                  query={'description':'scs_reference/history.csv、raw_scale4/history.csv、genept_scale4/history.csv；仅包含实际完成的 epoch。',
                         'sql':'python benchmarks/scs_paper/collect.py','engine':'bash','language':'bash',
                         'tables_used':['scs_reference/history.csv','raw_scale4/history.csv','genept_scale4/history.csv']}),
             dict(id='audit',label='6 个区域的原始计数／GenePT 映射完整性审计',path='runs/SCS_paper_benchmark_v1/integrity_audit.json',
                  query={'sql':'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python benchmarks/scs_paper/audit.py',
                         'engine':'bash','language':'bash','tables_used':['runs/SCS_paper_benchmark_v1/integrity_audit.json']}),
             dict(id='paper',label='SCS 作者开放稿与补充材料',href='https://pmc.ncbi.nlm.nih.gov/articles/PMC10312435/'),
             dict(id='protocol',label='锁定的实验协议与复现限制',path='benchmarks/scs_paper/README.md')]
    def md(i,text,source=None):
        b=dict(id=i,type='markdown',body=text)
        if source:b['sourceId']=source
        return b
    blocks=[md('title','# '+title),md('summary','## 当前结论仍需保留\n\n'+body,'results'),
            md('definitions','## 怎样理解这些分数\n\n方向准确率只在前景测试中心点计算；前景 balanced accuracy 为前景召回与背景特异度的平均。RNA 一致性是分割质量的间接指标，不是有真实边界的准确率。每个配对比较只纳入交集及两侧独有区均至少有 100 UMI 的共同核；不同配对行不能横向当成同一排行榜。','protocol'),
            md('learning','## 学习曲线能判断是否学到，但不能替代分割评估\n\n下图显示 2104 各方法每轮的前景方向验证准确率。所有方法计划训练 100 轮；曲线长度不足表示仍在训练，不表示其最终成绩。横轴按轮数比较，不代表相同训练耗时。','curve'),
            dict(id='learning-chart',type='chart',chartId='learning'),
            md('test','## 只报告已经完成的独立测试成绩\n\n验证集用于选择 checkpoint，测试集在选择后才评估。以下分母和状态可用于检查可比性；随机中心点的邻域仍可重叠，成绩是片内插值能力，不是跨组织泛化。','results'),
            dict(id='test-table',type='table',tableId='test'),
            md('segmentation','## 更高的方向准确率不一定带来更好的最终边界\n\n已完成的分割配对如下。Raw 4× 与 SCS 在 2104 的 RNA 相关性差距很小；SCS 与仅方向先验的另一组配对也没有显示明确的网络优势。先验对照保留 SCS 的前景预测，仅移除学习到的方向，因此不能称为完全不训练的基线。每行使用自己的共同合格样本，不能跨行比较绝对相关性。','results'),
            dict(id='rna-table',type='table',tableId='rna'),
            md('coverage','## GenePT 在肝脏丢失的输入计数不可忽略\n\n肝脏四区域的 GenePT 计数覆盖率只有约 59%–64%，seqFISH+ 两视野约 99.3%。分母是每个区域已选 2000 个基因的输入计数，不是全部测序 RNA。新增匹配覆盖的原始计数对照，避免将缺失基因的影响直接归因为 embedding 表达能力。','audit'),
            dict(id='coverage-chart',type='chart',chartId='coverage'),
            md('design','## 同数据与同训练预算的受控比较\n\n使用原文 Seq-scope 的 4 个肝脏区域，以及 seqFISH+ 实验 1 的 FOV 0、1。每个完整区域训练一个模型；不是从 ST19 挑容易的小块。各方法统一 2000 HVG、50 tokens、80/10/10 随机中心划分、100 epochs、batch 512、BF16、SDPA 和后处理。SCS 控制为 64 宽／8 层；新版为 256 宽／32 层，Muon 加辅助 AdamW。','protocol'),
            md('limitations','## 这是原文数据上的比较，不是论文分数的逐项复现\n\nSCS 基线采用经过等价验证的 PyTorch 移植及现有 Spateo 兼容层；batch、随机划分和精度不同于上游默认。当前只跑单随机种子。HVG 在整区域选择，邻域跨集合重叠。\n\nseqFISH+ 只选 2 个视野、30 个手工轮廓；公开 RNA 已按人工细胞整理。4 个原始像素一个网格只是约 0.4 µm，DAPI 的投影与归一化也明确记录，不能直接对照论文的 0.75 IoU。未完整标注全图，不报告全图 precision/AP；重叠的 ROI 边界像素统一忽略。','protocol'),
            md('next','## 下一步以完整分割和覆盖对照判断有效性\n\n等待各区域完成相同训练预算，再看优势是否在多个区域一致出现；对 seqFISH+ 检查一对一匹配 IoU 和 IoU≥0.5 的真值细胞召回率。当前 CPU 后处理与串行 GPU 训练队列都已启动。若提升只出现在方向 accuracy 或只来自覆盖差异，不把它判定为方法有效。\n\n仍待回答：GenePT 在高覆盖数据上是否保持效果？覆盖匹配后差距是否仍在？方向网络相对核位置先验是否真正改善边界？','protocol')]
    manifest=dict(version=1,title=title,generatedAt=now,blocks=blocks,sources=sources,
                  charts=[dict(id='learning',title='肝脏 2104 验证方向准确率',type='line',dataset='curve',sourceId='curve',
                               encodings={'x':{'field':'epoch','type':'quantitative'},'y':{'field':'validation_direction','type':'quantitative'},
                                          'color':{'field':'method','type':'nominal'}},valueFormat='percent',layout='full',
                               palette={'kind':'categorical'},legend={'position':'bottom'},labels={'values':'endpoints'}),
                          dict(id='coverage',title='GenePT 输入计数覆盖率',type='bar',dataset='coverage',sourceId='audit',
                               encodings={'x':{'field':'tile','type':'nominal'},'y':{'field':'umi_coverage','type':'quantitative'},
                                          'color':{'field':'dataset','type':'nominal'}},valueFormat='percent',layout='full',
                               palette={'kind':'categorical'},legend={'position':'bottom'},labels={'values':'all'})],
                  tables=[dict(id='test',title='2104 测试结果与状态',dataset='tests',sourceId='results',
                               columns=[{'field':k,'label':v,**({'format':'percent'} if k in ['direction_accuracy','foreground_bacc'] else {})} for k,v in [('method','方法'),('status','状态'),('direction_accuracy','方向准确率'),
                                        ('foreground_bacc','前景 balanced accuracy'),('selected_epoch','选中 epoch'),('test_foreground_n','测试前景点'),
                                        ('test_background_n','测试背景点')]],defaultSort={'field':'method','direction':'asc'}),
                          dict(id='rna',title='已完成的配对 RNA 一致性',dataset='pairs',sourceId='results',
                               columns=[{'field':k,'label':v} for k,v in [('tile','区域'),('first','方法 A'),('second','方法 B'),
                                        ('correlation_first','A 平均 r'),('correlation_second','B 平均 r'),('difference','A−B'),
                                        ('eligible_nuclei','共同合格核'),('unique_pairs','唯一细胞对')]],
                               defaultSort={'field':'tile','direction':'asc'})])
    status='ready' if full_segments==6 else 'partial'
    snapshot=dict(version=1,status=status,generatedAt=now,datasets=dict(curve=curve,tests=section2104,coverage=coverage,pairs=comparison))
    if status=='partial':snapshot['accessIssues']=[{'id':'pending-runs','scope':'benchmark','message':f'{completed}/18 次训练与 {full_segments}/6 个区域评估已完成；其余模型、IoU 和覆盖对照仍在队列中。'}]
    artifact=dict(surface='report',manifest=manifest,snapshot=snapshot,sources=sources)
    (root/'report_artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    # Supporting text handoff, also usable when the report renderer rejects
    # honest Python-file provenance. It is not advertised as rendered HTML.
    lines=['# SCS 原文数据 benchmark 状态',f'\n快照时间：{now}（UTC；不是自动刷新的监控页面）',
           f'\n已完成主要训练 {completed}/18；完整区域评估 {full_segments}/6。另有 1 次覆盖匹配消融。',
           '\n## 2104：已完成的独立测试结果',
           '\n所有方法相同的 100 epochs、随机 80/10/10 中心划分、batch 512、BF16。方向分母为前景测试点，不是全部点。',
           '\n| 方法 | 状态 | 方向准确率 | 前景 balanced accuracy | 选择 epoch |',
           '|---|---|---:|---:|---:|']
    for r in section2104:
        acc=f"{r['direction_accuracy']:.2%}" if r['direction_accuracy'] is not None else '—'
        ba=f"{r['foreground_bacc']:.2%}" if r['foreground_bacc'] is not None else '—'
        lines.append(f"| {r['method']} | {r['status']} | {acc} | {ba} | {r['selected_epoch'] or '—'} |")
    lines+=['\n## 最终分割：不能用方向准确率替代',
            '\n| 区域 | 方法 A | 方法 B | A 平均 r | B 平均 r | 配对核数 | 唯一细胞对 |',
            '|---|---|---|---:|---:|---:|---:|']
    for r in comparison:
        if r['correlation_first'] is not None:
            lines.append(f"| {r['tile']} | {r['first']} | {r['second']} | {r['correlation_first']:.4f} | {r['correlation_second']:.4f} | {r['eligible_nuclei']} | {r['unique_pairs']} |")
    lines+=['\n每一行仅比较该行共同合格核：交集和两侧独有区均 >=100 UMI。不同配对的合格群体不同，不可跨行排名。多个核可能匹配同一个细胞，不计算假定它们独立的显著性。',
            '\n## GenePT 覆盖限制',
            '\n肝脏 2000 基因输入中只有约 59%–64% 的计数可映射到现有字典；seqFISH+ 约 99.3%。肝脏主要缺失 mt-Co1、Gm26924、Mup3、Mup20 等。额外 raw_matched_scale4 对照同样屏蔽缺失基因，尚未完成时不能据此归因。',
            '\n## 第二数据集与结论边界',
            '\nseqFISH+ 实验 1 的 FOV 0/1 已完成准备，共 30 个人工轮廓；训练在肝脏队列之后串行执行。IoU 用一对一匹配，漏检真值计零。公开 RNA 已按人工细胞整理；网格尺度、DAPI 预处理和未完整标注全图的限制见实验协议。',
            '\n这是一套原文数据上的受控比较，不是原文分数的精确复现。当前单随机种子；随机中心测试允许邻域重叠。SCS 基线是 PyTorch 移植、现有 Spateo 兼容层及共同 batch/精度设置。没有足够证据宣称新方法总体有效或优于原文。',
            '\n## 可核查文件',
            '\n- 协议与运行入口：benchmarks/scs_paper/README.md',
            '- 数据与映射审计：runs/SCS_paper_benchmark_v1/integrity_audit.json',
            '- 机器可读状态：runs/SCS_paper_benchmark_v1/summary.json 与 summary.csv',
            '- 原始逐轮记录、配置、预测与完成标记：各区域的模型目录。',
            '\n原文：[SCS 开放稿及补充材料](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312435/)；[官方代码与肝脏数据](https://github.com/chenhcs/SCS)；[seqFISH+ 原始数据与人工轮廓](https://zenodo.org/records/2669683)。',
            '\n图表报告尚未生成：MCP 验证及官方便携 HTML 打包器均拒绝 Python 文件来源，要求 SQL 查询。本次保留真实 Python 来源，没有虚构 SQL 来绕过限制。上述数值文件不受影响。']
    (root/'benchmark_status.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['records','coverage','paired_rna_comparisons','segmentation_quality']}))


if __name__=='__main__':collect()
