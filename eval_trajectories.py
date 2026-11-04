import argparse
import json
from pathlib import Path

import numpy as np

from lib.trajectories import JOINT_SCHEMA, load_bundle


def align(source, target, scale=True, require_rank=0):
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 2:
        raise ValueError('Alignment needs matching N x 3 arrays with N >= 2')
    x, y = source - source.mean(0), target - target.mean(0)
    variance = np.sum(x * x) / len(x)
    covariance = y.T @ x / len(x)
    if require_rank and np.linalg.matrix_rank(covariance, tol=1e-10) < require_rank:
        raise ValueError('Degenerate trajectory cannot define this alignment')
    if scale and variance <= 1e-12:
        raise ValueError('Similarity scale is undefined for a static prediction')
    U, D, Vh = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        sign[-1] = -1
    rotation = (U * sign) @ Vh
    multiplier = np.dot(D, sign) / variance if scale else 1.0
    translation = target.mean(0) - multiplier * rotation @ source.mean(0)
    return multiplier, rotation, translation


def transform(points, alignment):
    scale, rotation, translation = alignment
    return scale * np.asarray(points) @ rotation.T + translation


def match_frames(pred, gt, timestamp_tolerance=0.02, allow_partial=False):
    ids, pi, gi = np.intersect1d(pred['frame_ids'], gt['frame_ids'], return_indices=True)
    if not len(ids):
        raise ValueError('Prediction and GT have no common original frame IDs')
    if not allow_partial and (len(ids) != len(pred['frame_ids']) or len(ids) != len(gt['frame_ids'])):
        raise ValueError('Timeline differs; use --allow-partial explicitly to score only common frames')
    if np.any(np.abs(pred['timestamps'][pi] - gt['timestamps'][gi]) > timestamp_tolerance):
        raise ValueError('Matched frame timestamps disagree; check time origin/FPS')
    keep = pred['valid'][pi] & gt['valid'][gi]
    return ids[keep], pi[keep], gi[keep]


def evaluate(pred_dir, gt_dir, protocol='emdb_legacy', timestamp_tolerance=0.02, allow_partial=False):
    if protocol not in ('emdb_legacy', 'generic'):
        raise ValueError('Unknown evaluation protocol')
    pm, pred = load_bundle(pred_dir)
    gm, gt = load_bundle(gt_dir)
    metrics, unavailable, coverage = {}, {}, {}
    if 'camera' in pred and 'camera' in gt:
        ids, pi, gi = match_frames(pred['camera'], gt['camera'], timestamp_tolerance, allow_partial)
        coverage['camera'] = {'matched_valid': len(ids), 'prediction_total': len(pred['camera']['frame_ids']),
                              'gt_total': len(gt['camera']['frame_ids'])}
        target = gt['camera']['c2w'][gi, :3, 3].astype(np.float64)
        source = pred['camera']['c2w'][pi, :3, 3].astype(np.float64)
        for key, scale in [('ate', True), ('ate_s', False)]:
            try:
                fitted = transform(source, align(source, target, scale, require_rank=2))
                metrics[key] = float(np.linalg.norm(target - fitted, axis=-1).mean())
            except ValueError as error:
                unavailable[key] = str(error)
    else:
        unavailable.update(ate='GT/pred camera unavailable', ate_s='GT/pred camera unavailable')
    if 'human' in pred and 'human' in gt:
        if pm['joint_schema'] != gm['joint_schema'] or pm['joint_schema'] != JOINT_SCHEMA:
            raise ValueError('Pred/GT evaluation joint definitions differ')
        if pm.get('assets', {}).get('joint_regressor_sha256') != gm.get('assets', {}).get('joint_regressor_sha256'):
            raise ValueError('Pred/GT joint-regressor asset identifiers differ')
        ids, pi, gi = match_frames(pred['human'], gt['human'], timestamp_tolerance, allow_partial)
        coverage['human'] = {'matched_valid': len(ids), 'prediction_total': len(pred['human']['frame_ids']),
                             'gt_total': len(gt['human']['frame_ids'])}
        key = 'eval_joints24_world' if protocol == 'emdb_legacy' else 'eval_joints24_world_raw'
        source_all = pred['human'].get(key, pred['human']['eval_joints24_world'])
        source = source_all[pi].astype(np.float32)
        target = gt['human']['eval_joints24_world'][gi].astype(np.float32)
        spans = [np.arange(len(ids))]
        if protocol == 'generic' and len(ids):
            step = np.median(np.diff(gt['human']['frame_ids'])) if len(gt['human']['frame_ids']) > 1 else 1
            spans = np.split(np.arange(len(ids)), np.flatnonzero(np.diff(ids) > step) + 1)
        errors = {'w_mpjpe': [], 'wa_mpjpe': []}
        for span in spans:
            for start in range(0, len(span), 100):
                chunk = span[start:start + 100]
                if len(chunk) < 2:
                    continue
                p, g = source[chunk], target[chunk]
                for name, anchor in [('w_mpjpe', slice(0, 2)), ('wa_mpjpe', slice(None))]:
                    try:
                        fit = align(p[anchor].reshape(-1, 3), g[anchor].reshape(-1, 3), scale=True)
                        errors[name].append(np.linalg.norm(g - transform(p, fit), axis=-1).mean(-1) * 1000)
                    except ValueError as error:
                        unavailable[name] = str(error)
        for name, values in errors.items():
            if name in unavailable:
                continue
            if not values:
                unavailable[name] = 'No valid chunks containing at least 2 frames'
                continue
            values = np.concatenate(values)
            metrics.update({name: float(values.mean()), '_sum_' + name: float(values.sum()),
                            '_cnt_' + name: int(len(values))})
        try:
            if len(target) < 2:
                raise ValueError('RTE needs at least 2 valid frames')
            target_root, pred_root = target[:, 0], source[:, 0]
            displacement = np.linalg.norm(np.diff(target_root, axis=0), axis=-1)
            if protocol == 'generic':
                step = np.median(np.diff(gt['human']['frame_ids']))
                displacement = displacement[np.diff(ids) <= step]
            path_length = float(displacement.sum())
            if path_length <= 1e-8:
                raise ValueError('RTE is undefined for a stationary GT root trajectory')
            fitted = transform(pred_root, align(pred_root, target_root, scale=False))
            values = np.linalg.norm(target_root - fitted, axis=-1) / path_length * 100
            metrics.update(rte=float(values.mean()), _sum_rte=float(values.sum()), _cnt_rte=int(len(values)))
        except ValueError as error:
            unavailable['rte'] = str(error)
    else:
        unavailable.update({key: 'GT/pred human unavailable' for key in ['w_mpjpe', 'wa_mpjpe', 'rte']})
    return {'protocol': protocol, 'metrics': metrics, 'unavailable': unavailable, 'coverage': coverage,
            'definitions': {'ate': 'Sim3-aligned camera center mean, m (not RMSE)',
                            'ate_s': 'SE3-aligned camera center mean, m',
                            'w_mpjpe': '100-frame chunks; first2-frame Sim3 alignment, mm',
                            'wa_mpjpe': '100-frame chunks; all-frame Sim3 alignment, mm',
                            'rte': 'rigid-aligned root error / GT path length, percent'},
            'prediction': str(Path(pred_dir).resolve()), 'ground_truth': str(Path(gt_dir).resolve())}


def main():
    parser = argparse.ArgumentParser(description='Evaluate published numerical trajectories; no models, SLAM or GPU required')
    parser.add_argument('--pred', type=Path, required=True)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', choices=['emdb_legacy', 'generic'], default='emdb_legacy')
    parser.add_argument('--timestamp-tolerance', type=float, default=.02)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if args.timestamp_tolerance < 0:
        parser.error('timestamp-tolerance must be nonnegative')
    result = evaluate(args.pred, args.gt, args.protocol, args.timestamp_tolerance, args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
