"""Create isolated run directories and preserve all previous evidence."""
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/Cellist_full_genept_v1'


def initialize():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside the allocated compute node')
    OUT.mkdir(exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    datasets = OUT / 'datasets'
    datasets.mkdir(exist_ok=True)
    source = ROOT / 'runs/SCS_stereo_whole_v1'
    sources = {'stereo_mouse_brain': source}
    sources.update({tile: source / 'liver' / tile for tile in ('2104', '2105', '2106', '2107')})
    for name, old in sources.items():
        dest = datasets / name
        dest.mkdir(exist_ok=True)
        for item in old.iterdir():
            if item.is_file() and item.suffix in ('.json', '.npy', '.npz', '.h5'):
                link = dest / item.name
                if not link.exists():
                    link.symlink_to(item.resolve())
        old_model = old / 'genept_all_scale4'
        new_model = dest / 'genept_all_scale4'
        if name == 'stereo_mouse_brain':
            marker = dest / 'checkpoint_imported.json'
            if not marker.exists():
                if new_model.exists():
                    raise RuntimeError('Incomplete checkpoint import: inspect before retry')
                new_model.mkdir()
                for item in old_model.iterdir():
                    if item.is_file() and (item.suffix == '.json' or item.name in ('latest.pt', 'best.pt')):
                        shutil.copy2(item, new_model / item.name)
                marker.write_text(json.dumps({'source': str(old_model), 'kind': 'copied_checkpoint_and_history',
                    'job': os.environ['SLURM_JOB_ID']}, indent=2))
        elif not new_model.exists():
            for marker in ('training_completed.json', 'inference_completed.json', 'whole_slide/completed.json'):
                if not (old_model / marker).exists():
                    raise RuntimeError(f'Incomplete candidate {old_model}/{marker}')
            new_model.symlink_to(old_model.resolve(), target_is_directory=True)
        manifest = {'source': str(old.resolve()), 'scope': 'full_measured_region',
                    'reuse': 'resume_copy' if name == 'stereo_mouse_brain' else 'read_only_completed_run',
                    'common_nuclei': str((dest / 'nuclei.npy').resolve()),
                    'resolution_um': 0.5 if name == 'stereo_mouse_brain' else 0.6,
                    'platform': 'barcoding'}
        (dest / 'comparison_input.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'launch.json').write_text(json.dumps({'job': os.environ['SLURM_JOB_ID'],
        'host': os.uname().nodename, 'scope': list(sources) + ['seqfish_rep1_fov0', 'seqfish_rep1_fov1'],
        'ST19_included': False, 'method': 'official_GenePT_all_mappable_genes_scale4',
        'gpu_plan': 'GPU 0 brain resume; GPU 1 seqFISH+; Cellist on CPU',
        'status': 'initialized'}, indent=2))
    print(OUT, flush=True)


if __name__ == '__main__':
    initialize()
