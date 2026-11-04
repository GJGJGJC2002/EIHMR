import argparse
import copy
import csv
import fcntl
import json
import logging
import os
import pickle
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from lib.pipeline.checkpoint import digest, load_trusted_pickle, read_npz, signature, write_json, write_npz
from lib.pipeline.core.data import CameraParams, PipelineData, SMPLParams
from lib.pipeline.pipeline import Pipeline, create_component
from lib.pipeline.video import detect_people, video_config
from lib.pipeline.runtime import runtime

with (ROOT / 'configs/emdb2_sequences.csv').open() as _stream:
    SPLITS = {row['seq']: row['split'] for row in csv.DictReader(_stream)}  # used for report grouping only
SEQUENCES = list(SPLITS)


def data_for_sequence(args, row):
    """EMDB images and annotations for one sequence (annotations are used for evaluation only)."""
    name, person = row['seq'], row['person']
    candidates = [args.dataset_root / person / name, args.dataset_root / person / person / name]
    paths = [path for path in candidates if (path / f'{person}_{name}_data.pkl').is_file()]
    if len(paths) != 1:
        raise ValueError(f'Expected one annotation directory for {person}/{name}: {paths}')
    annotation_path = paths[0] / f'{person}_{name}_data.pkl'
    with annotation_path.open('rb') as stream:
        annotations = pickle.load(stream)
    images = sorted((paths[0] / 'images').glob('*.jpg'))
    if not images:
        raise ValueError(f'No images: {paths[0]}')
    data = PipelineData(sequence_name=name, sequence_path=str(paths[0]), image_paths=[str(p) for p in images],
                        annotations=annotations, metadata={'dataset_type': 'emdb'})
    return data, digest(annotation_path)


def arguments():
    parser = argparse.ArgumentParser(description='End-to-end EMDB2 evaluation of the release pipeline: EIHMR and its '
                                                 'PromptHMR + Mask-DROID baseline from the same run')
    for name in ['dataset-root', 'asset-root', 'droid-root', 'prompthmr-root', 'detector-weights', 'output-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--body-model-root', type=Path,
                        help='SMPL/SMPL-X models (smplx/, smpl/); default <prompthmr-root>/data/body_models')
    parser.add_argument('--sequences', help='Comma-separated subset; default all EMDB2 sequences')
    parser.add_argument('--gpu-ids', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots-per-gpu', type=int, default=1)
    parser.add_argument('--worker', choices=SEQUENCES)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def config_for(args):
    return video_config(ROOT / 'configs/pipeline.yaml', {
        'prompthmr': args.prompthmr_root,
        'metric3d': args.prompthmr_root / 'pipeline/yvanyin_metric3d_main'}, interval=5)


def selected(args):
    names = args.sequences.split(',') if args.sequences else SEQUENCES
    unknown = sorted(set(names) - set(SEQUENCES))
    if unknown:
        raise ValueError(f'Unknown sequences: {unknown}')
    return names


def worker(args):
    """Run the full release pipeline on one sequence and evaluate it and its baseline.

    Pipeline inputs follow the common EMDB protocol: images, the annotated target-person box and the
    calibration K.  Everything else is predicted (YOLO for bystander masks, SAM, DROID, PromptHMR with
    ViTPose keypoints and SLAM rotations, Metric3D).  Annotations are otherwise used only for scoring.
    The baseline (PromptHMR + Mask-DROID) is the HPE checkpoint of the same run: first-pass masked DROID
    camera and PromptHMR body, before SA-LHMR, frontdelta and camera refine.
    """
    import torch
    from lib.camera.slam_utils import eval_slam
    from lib.utils.rotation_conversions import matrix_to_quaternion

    torch.set_num_threads(8)
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    seq = args.worker
    out = args.output_root / seq
    out.mkdir(parents=True, exist_ok=True)
    runtime(args)
    with (ROOT / 'configs/emdb2_sequences.csv').open() as stream:
        row = next(row for row in csv.DictReader(stream) if row['seq'] == seq)
    original, annotation_hash = data_for_sequence(args, row)
    annotations = original.annotations
    good = np.asarray(annotations['good_frames_mask'], dtype=bool)
    images = out / 'input/images'
    images.mkdir(parents=True, exist_ok=True)
    for path in original.image_paths:
        link = images / Path(path).name
        if not link.exists():
            link.symlink_to(Path(path))
        elif not link.is_symlink() or link.resolve() != Path(path).resolve():
            raise ValueError('Isolated input image path changed')
    paths = [str(images / Path(path).name) for path in original.image_paths]
    people = detect_people([Path(p) for p in original.image_paths], args.detector_weights, out / 'input')
    target = np.asarray(annotations['bboxes']['bboxes'], dtype=np.float32)
    score = np.ones((len(target), 1), dtype=np.float32)
    score[np.asarray(annotations['bboxes']['invalid_idxs'], dtype=np.int64)] = 0  # kept for HPE, not masked
    K = np.asarray(annotations['camera']['intrinsics'], dtype=np.float32)
    config = config_for(args)
    # component checkpoints add their own config and source hashes; scoring code is not part of the input
    identity = {'annotation_file': annotation_hash, 'detector': digest(args.detector_weights),
                'config': signature(config)}
    write_json(out / 'input_audit.json', {
        'protocol': 'EMDB: annotated target box and K; detected keypoints/bystanders; no GT camera or keypoints',
        'pipeline_annotations': False, 'num_frames': len(paths), 'invalid_box_frames': int((score == 0).sum()),
        'identity': identity})
    write_json(out / 'effective_config.json', config)
    data = PipelineData(sequence_name=seq, sequence_path=str(images.parent), image_paths=paths,
                        bboxes=np.concatenate([target, score], axis=1)[:, None],
                        camera_params=CameraParams(intrinsics=K), annotations=None, valid_frames_mask=None,
                        metadata={'inference_mode': 'video', 'dataset_type': 'video', 'output_dir': str(out),
                                  'stop_after': None, 'person_boxes': people, 'input_fingerprint': identity})
    result = Pipeline(config, out).execute(data)
    write_npz(out / 'camera.npz', result.camera_params.to_dict())
    write_npz(out / 'smpl.npz', result.smpl_params.to_dict())
    checkpoints = out / 'checkpoints' / seq
    baseline = load_trusted_pickle(checkpoints / 'hpe_data.pkl')
    baseline_camera = CameraParams.from_dict(baseline['camera_params'])
    baseline_smpl = SMPLParams.from_dict(baseline['smpl_params'])

    ext = annotations['camera']['extrinsics'][good]
    gt_R = ext[:, :3, :3].transpose(0, 2, 1)
    gt_T = -np.einsum('fij,fj->fi', gt_R, ext[:, :3, 3])
    gt_q = matrix_to_quaternion(torch.as_tensor(gt_R).float()).numpy()
    cameras = {'pass1_native': baseline_camera, 'final_human': result.camera_params}
    frontdelta = checkpoints / 'frontdelta_camera.npz'
    if frontdelta.exists():
        cameras['frontdelta'] = CameraParams.from_dict(read_npz(frontdelta))
    stages = {}
    for name, camera in cameras.items():
        R, T = np.asarray(camera.R), np.asarray(camera.T)
        q = matrix_to_quaternion(torch.as_tensor(R[good]).float()).numpy()
        trajectory = np.concatenate([T[good], q], axis=1)
        stages[name] = {key: float(eval_slam(trajectory.copy(), gt_T, gt_q, correct_scale=scale)[0]['mean'])
                        for key, scale in [('ate', True), ('ate_s', False)]}
    import yaml
    evaluator_config = next(c for c in yaml.safe_load((ROOT / 'configs/pipeline.yaml').read_text())['components']
                            if c['type'] == 'evaluation')
    evaluator_config.update(device='cuda')
    for name, camera, smpl in [('baseline_metrics.json', baseline_camera, baseline_smpl),
                               ('metrics.json', result.camera_params, result.smpl_params)]:
        evaluator = create_component(evaluator_config)
        evaluation = PipelineData(sequence_name=seq, annotations=annotations, valid_frames_mask=good,
                                  camera_params=camera, smpl_params=smpl,
                                  metadata={'hpe_stats': baseline['metadata'].get('hpe_stats', {}), 'dataset_type': 'emdb'})
        try:
            evaluation = evaluator(evaluation)
        finally:
            evaluator.cleanup()
        write_json(out / name, {key: float(value) for key, value in evaluation.metrics.items()})
    write_json(out / 'summary.json', {'success': True, 'route': result.metadata['second_pass_route'],
               'frontdelta': result.metadata['frontend_delta_slam'], 'metrics': stages,
               'baseline': 'PromptHMR + Mask-DROID (first pass, before SA-LHMR)',
               'finished_at': datetime.now().isoformat()})


HUMAN_KEYS = ['pa_mpjpe', 'mpjpe', 'pve', 'accel', 'w_mpjpe', 'wa_mpjpe', 'rte', 'jitter', 'fs']


def aggregate(values):
    """ATE/ATE-S: sequence mean; body metrics: frame-count weighted (same as run.aggregate)."""
    result = {key: float(np.mean([v[key] for v in values])) for key in ('ate', 'ate_s')}
    for key in HUMAN_KEYS:
        result[key] = sum(v[f'_sum_{key}'] for v in values) / sum(v[f'_cnt_{key}'] for v in values)
    return result


def report(args):
    names = selected(args)
    records, values, baselines, failed = [], {}, {}, []
    for seq in names:
        path = args.output_root / seq / 'summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            values[seq] = json.loads((args.output_root / seq / 'metrics.json').read_text())
            baselines[seq] = json.loads((args.output_root / seq / 'baseline_metrics.json').read_text())
            records.append({'sequence': seq, 'split': SPLITS[seq], 'pass1_travel_m': summary['route']['pass1_travel_m'],
                            'frontdelta_applied': bool(summary['frontdelta'].get('enabled')),
                            **{name + '_' + key: value[key] for name, value in summary['metrics'].items()
                               for key in ['ate', 'ate_s']},
                            **{'baseline_' + key: baselines[seq][key] for key in ['ate', 'ate_s', *HUMAN_KEYS]},
                            **{key: values[seq][key] for key in ['ate', 'ate_s', *HUMAN_KEYS]}})
        elif (args.output_root / seq / 'failure.json').exists():
            failed.append(seq)
    groups = []
    for group in ['Short', 'Medium', 'Long', 'All']:  # benchmark split, for reporting only
        chosen = [seq for seq in names if group == 'All' or SPLITS[seq] == group]
        done = [seq for seq in chosen if seq in values]
        for method, source in [('PromptHMR + Mask-DROID', baselines), ('EIHMR', values)]:
            item = {'split': group, 'method': method, 'completed': len(done), 'expected': len(chosen)}
            if done:
                item.update(aggregate([source[seq] for seq in done]))
            groups.append(item)
    for filename, rows in [('per_sequence_results.csv', records), ('group_summary.csv', groups)]:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        if fields:
            with (args.output_root / filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    write_json(args.output_root / 'run_status.json', {'completed': len(records), 'expected': len(names),
               'failed': failed, 'success': len(records) == len(names) and not failed,
               'finished_at': datetime.now().isoformat()})


def main():
    args = arguments()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.stop_after = None
    for source in (ROOT, args.dataset_root, args.asset_root, args.droid_root, args.body_model_root,
                   args.prompthmr_root):
        if source and (args.output_root == source or source in args.output_root.parents):
            raise ValueError(f'Output must not overwrite source data: {source}')
    if args.worker:
        directory = args.output_root / args.worker
        directory.mkdir(parents=True, exist_ok=True)
        worker_lock = (directory / 'worker.lock').open('a')
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            worker(args)
        except Exception:
            write_json(args.output_root / args.worker / 'failure.json', {'error': traceback.format_exc()})
            raise
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = (args.output_root / 'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'config': config_for(args), 'sequences': selected(args),
                'sources': {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob('*.py')},
                'paths': {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)}}
    identity = json.loads(json.dumps(identity))  # compare in stored form (e.g. int dict keys become strings)
    manifest = args.output_root / 'experiment_manifest.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())['identity'] != identity:
            raise ValueError('Existing experiment; --resume requires identical sources/configuration')
    else:
        write_json(manifest, {'identity': identity, 'created_at': datetime.now().isoformat(),
                   'scope': 'EMDB2 end-to-end release pipeline; baseline = PromptHMR + Mask-DROID of the same run',
                   'pipeline_uses_annotations': 'target box and K only'})
    gpus = [int(value) for value in args.gpu_ids.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 or g > 7 for g in gpus):
        raise ValueError('Unique GPU IDs 0 through 7 required')
    logs = args.output_root / 'logs'
    logs.mkdir(exist_ok=True)
    slots = [(gpu, slot) for slot in range(args.slots_per_gpu) for gpu in gpus]
    names = selected(args)

    def run(gpu, slot, shard):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        with (logs / f'gpu{gpu}_slot{slot}.log').open('a', buffering=1) as stream:
            for seq in shard:
                if (args.output_root / seq / 'summary.json').exists():
                    continue
                command = [sys.executable, str(Path(__file__).resolve()), *[v for v in sys.argv[1:] if v != '--resume'], '--worker', seq]
                code = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
                print(f'GPU{gpu} {seq}: exit={code}', flush=True)
                write_json(logs / f'{seq}_exit.json', {'gpu': gpu, 'returncode': code})
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(run, gpu, slot, names[i::len(slots)]) for i, (gpu, slot) in enumerate(slots)]
        for future in futures:
            future.result()
    report(args)
    return 0 if json.loads((args.output_root / 'run_status.json').read_text())['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
