import ast
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'tram'
SEEDS = [
    'lib/pipeline/core/data.py', 'lib/pipeline/core/component.py',
    'lib/pipeline/components/segmentation.py', 'lib/pipeline/components/slam.py',
    'lib/pipeline/components/hpe.py', 'lib/pipeline/components/depth_scene_refine.py',
    'lib/pipeline/components/adjacent_smpl_renderer.py',
    'lib/pipeline/components/evaluation.py',
    'lib/camera/frontend_delta.py', 'lib/camera/human_mesh_camera_refine.py',
    'lib/camera/masked_droid_slam.py', 'lib/camera/warmstart_droid.py',
    'lib/camera/est_gravity.py', 'lib/models/smpl.py',
]
SPECIAL = {
    'lib': '', 'lib.pipeline': '', 'lib.pipeline.core': '',
    'lib.pipeline.components': '', 'lib.pipeline.backends': '',
    'lib.models': '', 'lib.core': '', 'lib.utils': '', 'lib.vis': '',
    'lib.camera': ('from .masked_droid_slam import run_metric_slam, calibrate_intrinsics\n'
                   'from .warmstart_droid import run_metric_slam_warmstart, run_slam_warmstart\n'
                   'from .est_gravity import align_cam_to_world\n'),
}


def main():
    if (ROOT / 'source_manifest.json').exists():
        raise RuntimeError('Source snapshot already exists; refusing to overwrite edits')
    pending = list(SEEDS)
    copied = {}
    while pending:
        relative = pending.pop()
        if relative in copied:
            continue
        source = SOURCE / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        text = source.read_text()
        target = ROOT / relative
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        parts = Path(relative).with_suffix('').parts
        module = '.'.join(parts)
        package = module if parts[-1] == '__init__' else module.rsplit('.', 1)[0]
        for node in ast.walk(ast.parse(text)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    prefix = package.split('.')[:len(package.split('.')) - node.level + 1]
                    base = '.'.join(prefix + ([base] if base else []))
                modules = [base] + [base + '.' + alias.name for alias in node.names]
            for imported in modules:
                if not imported.startswith('lib.'):
                    continue
                candidate = SOURCE / (imported.replace('.', '/') + '.py')
                if candidate.is_file():
                    pending.append(str(candidate.relative_to(SOURCE)))
                else:
                    init = SOURCE / imported.replace('.', '/') / '__init__.py'
                    if init.is_file() and imported not in SPECIAL:
                        pending.append(str(init.relative_to(SOURCE)))
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            init = ROOT / '/'.join(parts[:i]) / '__init__.py'
            if parent in SPECIAL and not init.exists():
                init.write_text(SPECIAL[parent])
    shutil.copy2(SOURCE / 'LICENSE', ROOT / 'LICENSE')
    (ROOT / 'source_manifest.json').write_text(json.dumps({
        'source': 'tram working tree snapshot',
        'note': 'Hashes describe copied sources before release-specific edits; no historical commit claim.',
        'files': dict(sorted(copied.items())),
    }, indent=2) + '\n')
    print(f'Snapshotted {len(copied)} source files into {ROOT}')


if __name__ == '__main__':
    main()
