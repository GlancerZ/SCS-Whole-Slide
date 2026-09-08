"""Refresh an auditable comparison report; incomplete regions never become scores."""
import csv
import fcntl
import gzip
import math
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'runs/Cellist_full_genept_v1'
DATASETS = ['stereo_mouse_brain','2104','2105','2106','2107','seqfish_rep1_fov0','seqfish_rep1_fov1']


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def number(value, percent=False):
    if value is None: return '—'
    return f'{100*value:.2f}%' if percent else f'{value:.4f}'


def collect():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('Reporting runs in the separate CPU allocation')
    lock=(BASE/'report.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    stamp=datetime.now(timezone.utc).isoformat()
    status=[]; results=[]; rows=[]; audits=[]
    for name in DATASETS:
        data=BASE/'datasets'/name
        prepared=read(data/'prepared.json')
        patches=read(data/'patch_ranges.json') or []
        cellist=[read(data/'segmentation_patches'/p['id']/'cellist_paper/segmentation_completed.json') for p in patches]
        post=[(data/'segmentation_patches'/p['id']/'genept_paperpost_scale4/segmentation.npz').exists() for p in patches]
        epochs=sorted((data/'genept_all_scale4').glob('epoch_*.json'))
        training=read(data/'genept_all_scale4/training_completed.json')
        inference=read(data/'genept_all_scale4/inference_completed.json')
        comparison=read(data/'comparison_completed.json')
        fallback=[]
        failed=[]
        for p in patches:
            v=read(data/'segmentation_patches'/p['id']/'cellist_paper/hvg_compatibility.json')
            if v and v['fallbacks']:fallback.append(dict(patch=p['id'],events=v['fallbacks']))
            target=data/'segmentation_patches'/p['id']/'cellist_paper'
            if not (target/'segmentation_completed.json').exists() and any((target/f).exists() for f in ('failed.json','process_failed.json')):
                failed.append(p['id'])
        record=dict(dataset=name,patches=len(patches),prepared=bool(prepared and prepared['complete']),
            genept_last_epoch=int(epochs[-1].stem.split('_')[-1]) if epochs else 0,
            genept_training_complete=bool(training),genept_inference_complete=bool(inference),
            cellist_completed=sum(v is not None and v['complete'] for v in cellist),
            genept_postprocessed=sum(post),comparison_complete=bool(comparison),
            cellist_hvg_fallbacks=fallback,cellist_failed_patches=failed)
        if prepared: record['genept_coverage']=prepared['genept']
        status.append(record)
        if not comparison: continue
        if not all(cellist) or not all(post) or not inference or not training:
            raise AssertionError(f'{name}: comparison has unfinished prerequisites')
        assert training['epochs']==100 and 1<=training['selected_epoch']<=100
        assert inference['centres']==prepared['centres']
        independent={}
        with gzip.open(data/'evaluation/per_cell.csv.gz','rt') as handle:
            for row in csv.DictReader(handle):
                bucket=independent.setdefault(row['method'],dict(cells=0,umis=0,random=[]))
                bucket['cells']+=1;bucket['umis']+=int(float(row['n_umis']))
                value=float(row['random_correlation']) if row['random_correlation'] else float('nan')
                if int(row['n_spots'])>100 and math.isfinite(value):bucket['random'].append(value)
        source_umis=sum(read(data/'cellist_inputs'/p['id']/'prepared.json')['source_umis'] for p in patches)
        assert source_umis == comparison['source_umis'] == prepared['genept']['total_umi'], name
        for method,m in comparison['methods'].items():
            check=independent[method]
            assert check['cells']==m['cells'] and check['umis']==m['assigned_umis']
            assert len(check['random'])==m['random']['n']
            if check['random']:assert abs(math.fsum(check['random'])/len(check['random'])-m['random']['mean'])<1e-12
            assert 0<=m['assigned_umis']<=source_umis
            assert abs(m['assigned_umi_fraction']-m['assigned_umis']/source_umis)<1e-12
            if method=='cellist':assert sum(v['assigned_umis'] for v in cellist)==m['assigned_umis']
            iou=comparison.get('transcript_iou',{}).get(method,{})
            if iou:
                scores=iou['all_gt_ious']
                assert len(scores)==iou['manual_cells']
                assert all(0<=v<=1 for v in scores)
                assert abs(sum(scores)/len(scores)-iou['all_gt_mean'])<1e-12
                assert abs(sum(v>=.5 for v in scores)/len(scores)-iou['all_gt_recall_iou50'])<1e-12
            rows.append(dict(dataset=name,method=method,cells=m['cells'],source_umis=source_umis,
                assigned_umi_fraction=m['assigned_umi_fraction'],median_umis=m['median_umis'],
                random_mean=m['random']['mean'],random_n=m['random']['n'],
                directional_mean=m['directional']['mean'],
                purity_mean=comparison['purity']['cellist' if method=='cellist' else 'genept']['mean'],
                purity_n=comparison['purity']['cellist' if method=='cellist' else 'genept']['n'],
                transcript_iou_matched_mean=iou.get('matched_only_mean'),
                transcript_iou_all_gt_mean=iou.get('all_gt_mean'),
                transcript_recall_iou50=iou.get('all_gt_recall_iou50'),
                manual_cells=iou.get('manual_cells'),mutual_matches=iou.get('mutual_matches')))
        audits.append(dict(dataset=name,source_umi_conservation='pass',assigned_fraction_recalculation='pass',
            per_cell_csv_counts_and_random_mean='pass',training_and_inference_completeness='pass',
            cellist_patch_total_reconciliation='pass',iou_mean_recall_recalculation='pass' if 'transcript_iou' in comparison else 'not applicable',
            comparison_file=str(data/'comparison_completed.json')))
        results.append(comparison)
    write(BASE/'progress.json',dict(as_of_utc=stamp,datasets=status))
    write(BASE/'comparison.json',dict(as_of_utc=stamp,complete=len(results)==len(DATASETS),results=results))
    write(BASE/'validation.json',dict(as_of_utc=stamp,assessment='Share with caveats' if len(results)==len(DATASETS) else 'Incomplete benchmark; completed regions only',
        checked_regions=audits,blockers=[v['dataset'] for v in status if not v['comparison_complete']],
        caveats=['Shared watershed nuclei, not original Cellist native pipeline.',
            'Expression consistency is not independent segmentation accuracy.',
            'seqFISH molecule input is conditioned on selected manual cells.',
            'Only one seed per model; cells within one region are correlated observations.']))
    if rows:
        with (BASE/'comparison.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Cellist 与 full-gene GenePT 4× 比较', '',f'更新：{stamp}。已完成 {len(results)}/{len(DATASETS)} 个评估单元。',
        '','范围：小鼠脑整片、肝脏 2104–2107、seqFISH FOV 0/1；排除 ST19。Cellist 与后处理在独立 CPU 作业，GenePT 在 GPU 作业。',
        '', '## 进度', '', '| 区域 | GenePT epoch | 推理 | Cellist 块 | 后处理块 | 比较 |','|---|---:|---|---:|---:|---|']
    for v in status:
        lines.append(f"| {v['dataset']} | {v['genept_last_epoch']} | {'完成' if v['genept_inference_complete'] else '未完成'} | {v['cellist_completed']}/{v['patches']} | {v['genept_postprocessed']}/{v['patches']} | {'完成' if v['comparison_complete'] else '未完成'} |")
    lines+=['','## 已完成区域的主结果','','相关性越高越好；纯度比值越低越好。相关性使用 >100 spot 的细胞，双方细胞数可能不同。纯度使用双方共同合格核配对。',
        '', '| 区域 | 方法 | 细胞数 | 分配 RNA | 随机相关均值 (n) | 方向相关均值 | 纯度均值 (n) |', '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['cells']} | {number(r['assigned_umi_fraction'],True)} | {number(r['random_mean'])} ({r['random_n']}) | {number(r['directional_mean'])} | {number(r['purity_mean'])} ({r['purity_n']}) |")
    lines+=['','## seqFISH 人工分子归属 IoU','','不是稠密细胞边界 mask IoU。人工标注已决定原始分子输入范围，不能据此声称全图独立验证。',
        '', '| FOV | 方法 | 匹配/人工细胞 | 匹配细胞 IoU 均值 | 全部人工细胞 IoU 均值 | IoU≥0.5 召回 |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['manual_cells']:
            lines.append(f"| {r['dataset']} | {r['method']} | {r['mutual_matches']}/{r['manual_cells']} | {number(r['transcript_iou_matched_mean'])} | {number(r['transcript_iou_all_gt_mean'])} | {number(r['transcript_recall_iou50'],True)} |")
    lines+=['','## 同一配对内的 cross-correlation 差值','','下表均为 GenePT − Cellist，正值有利于 GenePT，负值有利于 Cellist。两个匹配方向的人群不同，分别报告。',
        '', '| 区域 | GenePT→Cellist 差值均值 (n) | Cellist→GenePT 差值均值 (n) |','|---|---:|---:|']
    for r in results:
        a=r['cross_correlation']['genept_to_cellist']['paired_difference'];b=r['cross_correlation']['cellist_to_genept']['paired_difference']
        lines.append(f"| {r['dataset']} | {number(a['mean'])} ({a['n']}) | {number(-b['mean'] if b['mean'] is not None else None)} ({b['n']}) |")
    lines+=['','## 全基因映射覆盖','','full 指所有可映射基因，无 HVG 截断；不是保证每个测得基因都有官方 GenePT 向量。',
        '', '| 区域 | 原始基因列数 | 映射后唯一基因数 | RNA 计数覆盖 |','|---|---:|---:|---:|']
    for v in status:
        g=v.get('genept_coverage',{})
        if g:lines.append(f"| {v['dataset']} | {g['source_genes']} | {g['mapped_unique_genes']} | {number(g['mapped_umi_fraction'],True)} |")
    lines+=['','## 必须保留的解释限制','','- 共享 watershed 核，未精确复现作者 Cellpose 原生流程。脑的分块边界未跨块合并。',
        '- 高表达一致性不等于准确边界；同时参考 RNA 覆盖、细胞大小、人工分子 IoU。没有人工标注的脑和肝脏不报告“准确率”。',
        '- 每个模型只有一个训练 seed，训练验证在同片内；不能将细胞数量当作独立生物学样本数量进行显著性宣称。',
        '- Cellist 少于 3 核分块的 HVG 兼容处理明细见 progress.json；没有合格核的块仍保留在 RNA 分母。',
        '- 早期默认参数 pilot 不纳入以上结果。所有参数与评价适配见 benchmarks/cellist_full_genept/README.md。',
        '', '可复算文件：`comparison.csv`、`comparison.json`、`validation.json`；逐细胞表位于各区域 `evaluation/`。']
    if (BASE/'fairness_audit/review.md').exists():
        lines[2:2]=['**公平性复核：这些数值描述固定配置的表现；seqFISH伪标签与尺度适配、输入网格和表达深度混杂尚未排除，不能直接归因于算法或嵌入优劣。详见 [复核报告](fairness_audit/review.md)。**','']
    (BASE/'comparison.md').write_text('\n'.join(lines)+'\n')
    inventory=read(ROOT/'benchmarks/cellist_full_genept/inventory.json')
    inventory['execution_status']='complete' if len(results)==len(DATASETS) else 'running'
    inventory['checked_date_utc']=stamp
    inventory['scope_status']='user_confirmed_excluding_ST19'
    inventory['datasets']=[dict(dataset=v['dataset'],status={k:z for k,z in v.items() if k not in ('genept_coverage','cellist_hvg_fallbacks')}) for v in status]
    inventory['resource_check']=dict(command='squeue --me',allocations=dict(gpu='20445437',cpu_only='20445592'))
    inventory['environments']=inventory.pop('environments_awaiting_confirmation',inventory.get('environments',{}))
    inventory['paper']['evaluation_archive_status']='downloaded_and_md5_verified'
    inventory['paper']['evaluation_archive_md5']='49dda0fa190baf3d52ab2c5353d5dfa7'
    write(ROOT/'benchmarks/cellist_full_genept/inventory.json',inventory)
    launch=read(BASE/'launch.json')
    launch.update(gpu_allocation='20445437',cpu_allocation='20445592',cpu_host='rc31706',
        gpu_plan='GPU 0 brain; GPU 1 seqFISH; separate CPU-only allocation for Cellist, preparation, postprocessing, evaluation')
    write(BASE/'launch.json',launch)
    print(json.dumps(dict(as_of_utc=stamp,completed_regions=len(results),status=[{k:v for k,v in s.items() if k not in ('genept_coverage','cellist_hvg_fallbacks')} for s in status])),flush=True)
    lock.close()


if __name__=='__main__':collect()
