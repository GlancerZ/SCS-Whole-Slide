"""Capture implementation/environment evidence without altering running models."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/Cellist_full_genept_v1'


def capture(environment):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise RuntimeError('Use separate CPU allocation for provenance capture')
    out=BASE/'provenance';out.mkdir(exist_ok=True)
    distributions={}
    for name in ['numpy','scipy','pandas','anndata','scanpy','scikit-misc','scikit-image','torch','h5py','Cellist','spateo-release']:
        try:distributions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:pass
    info=dict(captured_utc=datetime.now(timezone.utc).isoformat(),environment=environment,python=platform.python_version(),
        executable=os.sys.executable,host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],packages=distributions)
    (out/f'environment_{environment}.json').write_text(json.dumps(info,indent=2)+'\n')
    hashes={}
    files=list((ROOT/'benchmarks/cellist_full_genept').glob('*.py'))+list((ROOT/'benchmarks/cellist_full_genept').glob('*.sh'))
    files += [ROOT/p for p in ['benchmarks/scs_paper/stereo_whole_train.py','benchmarks/scs_paper/stereo_whole_prepare.py',
        'benchmarks/scs_paper/stereo_whole_liver.py','benchmarks/scs_paper/evaluate.py','SCS/src/postprocessing.py',
        'optimizations/scs_streaming/torch_model.py','optimizations/scs_streaming/genept.py','scripts/cellist_cpu_stage.py']]
    files += list((ROOT/'tools/Cellist/src/Cellist').glob('*.py'))
    for path in files:
        rel=path.relative_to(ROOT);target=out/'code'/rel
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'code_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    diff=subprocess.run(['git','-C',str(ROOT/'tools/Cellist'),'diff','--','src/Cellist'],capture_output=True,text=True,check=True)
    (out/'Cellist_local_changes.patch').write_text(diff.stdout)
    checks={}
    for tile in ['2104','2105','2106','2107']:
        old=json.loads((ROOT/'runs/SCS_paper_benchmark_v1'/tile/'all_genes.json').read_text())
        current=json.loads((BASE/'datasets'/tile/'genes.json').read_text())
        assert old==current,f'{tile}: gene column order differs'
        config=json.loads((BASE/'datasets'/tile/'genept_all_scale4/config.json').read_text())
        expected=json.loads((BASE/'datasets'/tile/'prepared.json').read_text())
        assert config['genept']==expected['genept']
        assert config['neighbours_sha256']==expected['neighbors_sha256']
        assert config['split_sha256']==expected['split_sha256']
        checks[tile]=dict(gene_column_order='identical',model_genept_and_graph_split_fingerprints='identical',
            model_outputs_reused_read_only=(BASE/'datasets'/tile/'genept_all_scale4').is_symlink())
    (out/'liver_reuse_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(dict(environment=environment,code_files=len(hashes),liver_reuse_audit='pass')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--environment',required=True);args=p.parse_args();capture(args.environment)
