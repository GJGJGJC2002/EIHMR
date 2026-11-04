import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from lib.pipeline.checkpoint import read_npz
from lib.pipeline.core.data import CameraParams, PipelineData, SMPLParams
from lib.trajectories import JOINT_SCHEMA, _hash, export_prediction, poses, save_bundle


ROOT = Path(__file__).resolve().parent


def export_emdb_gt(annotation, directory, asset_root, fps, batch_size=128, device='cpu'):
    import smplx
    import torch
    from lib.gvhmr import ASSETS
    n = annotation['n_frames']
    model = smplx.SMPL(str(Path(asset_root) / 'smpl'), gender=annotation['gender'],
                       create_body_pose=False, create_betas=False, create_global_orient=False,
                       create_transl=False).to(device).eval()
    regressor_path = ASSETS / 'smpl_neutral_J_regressor.pt'
    regressor = torch.load(regressor_path, weights_only=True, map_location=device)
    smpl = annotation['smpl']
    joints = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
            result = model(body_pose=tensor(smpl['poses_body'][start:end]),
                           global_orient=tensor(smpl['poses_root'][start:end]),
                           transl=tensor(smpl['trans'][start:end]),
                           betas=tensor(np.broadcast_to(smpl['betas'].reshape(1, -1), (end - start, 10)).copy()))
            joints.append((regressor @ result.vertices).cpu().numpy())
    extrinsics = annotation['camera']['extrinsics']
    R = extrinsics[:, :3, :3].transpose(0, 2, 1)
    T = -np.einsum('fij,fj->fi', R, extrinsics[:, :3, 3])
    ids = np.arange(n, dtype=np.int64)
    valid = np.asarray(annotation.get('good_frames_mask', np.ones(n, dtype=bool)), dtype=bool)
    common = {'frame_ids': ids, 'timestamps': ids / fps, 'valid': valid}
    return save_bundle(directory, dict(common, c2w=poses(R, T), K=annotation['camera']['intrinsics']),
                       dict(common, eval_joints24_world=np.concatenate(joints)),
                       {'kind': 'ground_truth', 'world_frame': 'EMDB_annotation_world',
                        'joint_schema': JOINT_SCHEMA, 'postprocess': 'none',
                        'assets': {'joint_regressor_sha256': _hash(regressor_path)}})


def main():
    parser = argparse.ArgumentParser(description='Materialize publishable camera/body trajectories from completed predictions; no inference rerun')
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--body-model-root', type=Path, required=True,
                        help='Directory with smplx/SMPLX_NEUTRAL.npz and smpl/SMPL_NEUTRAL.pkl')
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--sequences', nargs='*')
    parser.add_argument('--fps', type=float, required=True, help='Explicit original benchmark/video FPS; not guessed')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--vertices', action='store_true')
    parser.add_argument('--export-emdb-gt', action='store_true')
    args = parser.parse_args()
    if args.fps <= 0 or args.batch_size < 1:
        parser.error('fps and batch-size must be positive')
    if args.export_emdb_gt and (not args.asset_root or not args.dataset_root):
        parser.error('EMDB GT conversion requires --asset-root and --dataset-root')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    from lib.gvhmr import set_body_model_root
    set_body_model_root(args.body_model_root)
    for source in [args.results_root, args.body_model_root, args.dataset_root, args.asset_root, ROOT]:
        if source and (args.output_root == source or source in args.output_root.parents):
            raise ValueError(f'Output must not overwrite prediction/source data: {source}')
    candidates = sorted(p.parent for p in args.results_root.glob('*/smpl.npz'))
    if args.sequences:
        missing = set(args.sequences) - {p.name for p in candidates}
        if missing:
            raise ValueError(f'Unknown prediction sequences: {missing}')
        candidates = [p for p in candidates if p.name in args.sequences]
    if not candidates:
        raise ValueError('No completed camera.npz/smpl.npz predictions found')
    with (ROOT / 'configs/emdb2_sequences.csv').open() as stream:
        benchmark = {row['seq']: row for row in csv.DictReader(stream)}
    reports = []
    for sequence in candidates:
        camera = CameraParams.from_dict(read_npz(sequence / 'camera.npz'))
        smpl = SMPLParams.from_dict(read_npz(sequence / 'smpl.npz'))
        n = len(camera.T)
        ids = np.arange(n, dtype=np.int64)
        annotation, valid = None, None
        if args.dataset_root:
            row = benchmark.get(sequence.name)
            if row is None:
                raise ValueError('Dataset adapter supports the recorded EMDB2 sequence list only')
            dirs = [args.dataset_root / row['person'] / sequence.name,
                    args.dataset_root / row['person'] / row['person'] / sequence.name]
            files = [p / f'{row["person"]}_{sequence.name}_data.pkl' for p in dirs
                     if (p / f'{row["person"]}_{sequence.name}_data.pkl').is_file()]
            if len(files) != 1:
                raise ValueError(f'Expected one trusted EMDB annotation: {sequence.name}')
            with files[0].open('rb') as stream:
                annotation = pickle.load(stream)
            if annotation['n_frames'] != n:
                raise ValueError('Annotation and final prediction frame counts differ')
            valid = annotation.get('good_frames_mask')
        data = PipelineData(sequence_name=sequence.name, camera_params=camera, smpl_params=smpl)
        output = args.output_root / sequence.name
        export_prediction(data, output / 'prediction',
                          {'frame_ids': ids, 'timestamps': ids / args.fps}, valid,
                          args.batch_size, args.device, args.vertices)
        report = {'sequence': sequence.name, 'frames': n, 'prediction': str(output / 'prediction')}
        if args.export_emdb_gt:
            export_emdb_gt(annotation, output / 'ground_truth', args.asset_root,
                           args.fps, args.batch_size, args.device)
            from eval_trajectories import evaluate
            body_frames = len(smpl.global_orient_c)
            evaluation = evaluate(output / 'prediction', output / 'ground_truth', allow_partial=body_frames != n)
            (output / 'evaluation.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False) + '\n')
            metrics = evaluation['metrics']
            saved = sequence / 'metrics.json'
            if saved.exists():
                expected = json.loads(saved.read_text())
                report['difference_from_original_evaluator'] = {key: metrics[key] - expected[key]
                    for key in ['ate', 'ate_s', 'w_mpjpe', 'wa_mpjpe', 'rte'] if key in metrics and key in expected}
            report['evaluation'] = evaluation
        reports.append(report)
        print(sequence.name, 'exported', flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'export_report.json').write_text(json.dumps(reports, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
