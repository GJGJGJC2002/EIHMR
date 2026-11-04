import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lib.camera import frontend_delta as frontend
from lib.camera.human_mesh_camera_refine import _so3_exp
from lib.pipeline.checkpoint import CameraCheckpoint, apply_camera, read_npz
from lib.pipeline.components.camera_refine import CameraRefineComponent
from lib.pipeline.core.data import CameraParams, PipelineData


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from eval_emdb2 import HUMAN_KEYS, aggregate  # noqa: E402


def data(directory):
    return PipelineData(sequence_name='test',
                        camera_params=CameraParams(R=np.repeat(np.eye(3)[None], 7, axis=0).astype(np.float32),
                                                   T=np.zeros((7, 3), dtype=np.float32)),
                        metadata={'output_dir': directory, 'input_fingerprint': {'input': 'test'},
                                  'oar_window_mesh_observations': [{'center': 0}]})


class ReleasePipelineTest(unittest.TestCase):
    def test_config_has_one_pipeline_without_benchmark_splits(self):
        import yaml
        config = yaml.safe_load((ROOT / 'configs/pipeline.yaml').read_text())
        for key in ('split_policy', 'target', 'target_rounding_digits'):
            self.assertNotIn(key, config)
        components = {c['type']: c for c in config['components']}
        self.assertEqual(sum(c['type'] == 'camera_refine' for c in config['components']), 1)
        self.assertNotIn('enabled', components['camera_refine']['frontend'])  # frontdelta on for every video
        self.assertNotIn('frontend_delta_slam', components)
        self.assertNotIn('human_mesh_camera_refine', components)

    def test_numpy_masks_converted_before_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            current = data(directory)
            render = Path(directory) / 'render'
            render.mkdir()
            (render / '00000.jpg').write_bytes(b'test')
            current.image_paths = [str(render / '00000.jpg')]
            current.camera_params.intrinsics = np.eye(3, dtype=np.float32)
            current.masks = np.zeros((1, 2, 2), dtype=np.float32)
            current.metadata['slam_keyframe_info'] = {'keyframe_tstamps': [0]}
            current.metadata['adjacent_render_info'] = {'output_dir': str(render)}
            current.metadata['stop_after'] = 'frontdelta'
            def refine(**kwargs):
                self.assertIsInstance(kwargs['masks'], torch.Tensor)
                self.assertEqual(kwargs['masks'].device.type, 'cpu')
                return kwargs['cam_R'], kwargs['cam_T'], {'enabled': True}
            with patch('lib.camera.frontend_delta.refine_camera_with_frontend_delta', side_effect=refine):
                CameraRefineComponent('camera_refine', {'device': 'cpu', 'frontend': {'enabled': True}, 'human': {}})(current)
            self.assertTrue(current.should_stop)

    def test_checkpoint_replaces_existing_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            original = data(directory)
            apply_camera(original, original.camera_params.R, original.camera_params.T + .1)
            checkpoint = CameraCheckpoint(directory, 'frontdelta', {'input': 'same'})
            checkpoint.save(original, {'enabled': True})
            fresh = data(directory)
            self.assertEqual(checkpoint.load(fresh), {'enabled': True})
            np.testing.assert_allclose(fresh.camera_params.T, .1)
            np.testing.assert_allclose(fresh.camera_params.world_T, .1)

    def test_stale_and_corrupt_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = data(directory)
            checkpoint = CameraCheckpoint(directory, 'frontdelta', {'input': 'same'})
            checkpoint.save(original, {'enabled': True})
            with self.assertRaises(ValueError):
                CameraCheckpoint(directory, 'frontdelta', {'input': 'changed'}).load(original)
            checkpoint.camera_path.write_bytes(b'incomplete')
            with self.assertRaises(ValueError):
                checkpoint.load(original)

    def test_pause_resume_matches_uninterrupted(self):
        config = {'device': 'cpu', 'frontend': {'enabled': False}, 'human': {}}
        def refine(R, T, observations, **kwargs):
            return torch.as_tensor(R), torch.as_tensor(T) + .02, {'enabled': True, 'accepted': True}
        with tempfile.TemporaryDirectory() as directory, patch(
            'lib.camera.human_mesh_camera_refine.refine_camera_with_window_mesh_consistency', side_effect=refine
        ) as mock:
            uninterrupted = data(directory + '/one')
            component = CameraRefineComponent('camera_refine', config)
            uninterrupted = component(uninterrupted)
            paused = data(directory + '/two')
            paused.metadata['stop_after'] = 'frontdelta'
            paused = component(paused)
            self.assertTrue(paused.should_stop)
            self.assertEqual(mock.call_count, 1)
            resumed = component(data(directory + '/two'))
            np.testing.assert_array_equal(resumed.camera_params.T, uninterrupted.camera_params.T)
            self.assertEqual(mock.call_count, 2)
            again = component(data(directory + '/two'))
            np.testing.assert_array_equal(again.camera_params.T, uninterrupted.camera_params.T)
            self.assertEqual(mock.call_count, 2)

    def test_weighted_aggregation_not_sequence_mean(self):
        def metrics(ate, frames, value):
            result = {'ate': ate, 'ate_s': 2 * ate}
            for key in HUMAN_KEYS:
                result.update({f'_sum_{key}': value * frames, f'_cnt_{key}': frames})
            return result
        result = aggregate([metrics(.1, 1, 10.), metrics(.3, 9, 20.)])
        self.assertAlmostEqual(result['ate'], .2)  # sequence mean
        self.assertAlmostEqual(result['w_mpjpe'], 19.)  # frame-weighted
        self.assertAlmostEqual(result['rte'], 19.)

    def test_frontdelta_guard_rejects_raw_excess_and_nonfinite(self):
        dT = np.zeros((4, 3), dtype=np.float32)
        dR = np.zeros((4, 3), dtype=np.float32)
        self.assertIsNone(frontend._guard_requested_delta(dT, dR, .05, 1, True)['reason'])
        dT[0, 0] = .1
        self.assertEqual(frontend._guard_requested_delta(dT, dR, .05, 1, True)['reason'], 'requested_delta_exceeds_limits')
        dT[:] = 0
        dR[2, 0] = np.radians(2)
        self.assertEqual(frontend._guard_requested_delta(dT, dR, .05, 1, True)['reason'], 'requested_delta_exceeds_limits')
        self.assertIsNone(frontend._guard_requested_delta(dT, dR, .05, 1, False)['reason'])
        dT[1, 0] = np.nan
        self.assertEqual(frontend._guard_requested_delta(dT, dR, .05, 1, True)['reason'], 'nonfinite_frontend_delta')

    def test_frontdelta_failure_falls_back_and_survives_pause_resume(self):
        config = {'device': 'cpu', 'frontend': {'enabled': True}, 'human': {}}
        with tempfile.TemporaryDirectory() as directory:
            current = data(directory)
            current.metadata.update(stop_after='frontdelta',
                                    slam_keyframe_info={'keyframe_tstamps': [0], 'keyframe_poses_se3': np.zeros((1, 7))},
                                    inference_mode='video')
            current.camera_params.intrinsics = np.eye(3)
            current.image_paths = [f'{directory}/00000.jpg']
            component = CameraRefineComponent('camera_refine', config)
            paused = component(current)
            self.assertTrue(paused.should_stop)
            np.testing.assert_array_equal(paused.camera_params.T, 0)
            self.assertTrue(paused.metadata['frontend_delta_slam']['fallback_to_baseline'])
            resumed = data(directory)
            resumed.camera_params.intrinsics = np.eye(3)
            def human(R, T, observations, **kwargs):
                np.testing.assert_array_equal(T, 0)
                return R, torch.as_tensor(T) + .02, {'enabled': True, 'accepted': True}
            with patch('lib.camera.human_mesh_camera_refine.refine_camera_with_window_mesh_consistency', side_effect=human):
                result = component(resumed)
            np.testing.assert_allclose(result.camera_params.T, .02)
            self.assertEqual(result.metadata['frontend_delta_slam']['reason'], 'missing_keyframes_or_rendered_images')

    def test_uniform_route_warm_starts_the_same_frontdelta_and_resumes_human(self):
        route = {'mode': 'uniform_keyframes', 'forced_keyframes': list(range(0, 41, 5)), 'keyframe_stride': 5,
                 'local_metric_scale': {'strength': .5}}
        info = {'keyframe_tstamps': np.array(route['forced_keyframes']), 'keyframe_poses_se3': np.zeros((9, 7))}
        config = {'device': 'cpu', 'frontend': {'enabled': True, 'max_translation_delta': .15,
                                                'max_rotation_delta_deg': 2.}, 'human': {}}
        with tempfile.TemporaryDirectory() as directory:
            def fresh():
                current = data(directory)
                current.camera_params.R = np.repeat(np.eye(3, dtype=np.float32)[None], 41, axis=0)
                current.camera_params.T = np.zeros((41, 3), dtype=np.float32)
                current.camera_params.intrinsics = np.eye(3)
                current.image_paths = [f'{directory}/{i:05d}.jpg' for i in range(41)]
                current.metadata.update(inference_mode='video', second_pass_route=route,
                                        slam_keyframe_info={'keyframe_tstamps': [0, 40]},
                                        adjacent_render_info={'output_dir': directory})
                return current
            for t in route['forced_keyframes']:
                (Path(directory) / f'{t:05d}.jpg').write_bytes(b'fixture')
            component = CameraRefineComponent('camera_refine', config)
            current = fresh()
            current.metadata['stop_after'] = 'frontdelta'
            def uniform(**kwargs):
                self.assertEqual(kwargs['forced_keyframes'], route['forced_keyframes'])
                self.assertEqual(kwargs['local_scale'], route['local_metric_scale'])
                self.assertEqual((kwargs['max_translation_delta'], kwargs['max_rotation_delta_deg']), (.15, 2.))
                return kwargs['cam_R'], kwargs['cam_T'] + .01, info, {'enabled': True}
            def delta(**kwargs):
                self.assertIs(kwargs['keyframe_info'], info)
                self.assertEqual(kwargs['max_translation_delta'], .15)
                np.testing.assert_allclose(kwargs['cam_T'], .01)
                return kwargs['cam_R'], torch.as_tensor(kwargs['cam_T']) + .005, {'enabled': True}
            with patch('lib.camera.frontend_delta.pull_back_uniform_keyframes', side_effect=uniform) as u, patch(
                    'lib.camera.frontend_delta.refine_camera_with_frontend_delta', side_effect=delta) as d:
                paused = component(current)
                self.assertTrue(paused.should_stop)
                np.testing.assert_allclose(paused.camera_params.T, .015)
                self.assertTrue(paused.metadata['frontend_delta_slam']['uniform_keyframes']['enabled'])
                def human(R, T, observations, **kwargs):
                    np.testing.assert_allclose(T, .015)
                    return R, T + .02, {'enabled': True, 'accepted': True}
                with patch('lib.camera.human_mesh_camera_refine.refine_camera_with_window_mesh_consistency',
                           side_effect=human) as h:
                    result = component(fresh())
                    np.testing.assert_allclose(result.camera_params.T, .035)
                    h.assert_called_once()
                u.assert_called_once()
                d.assert_called_once()

    def test_uniform_keyframe_pass_is_pulled_back_with_clamped_deltas(self):
        import sys
        import types
        n = 41
        R0 = np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0)
        T0 = np.stack([np.linspace(0, 4, n), np.sin(np.linspace(0, 3, n)), np.zeros(n)], axis=1).astype(np.float32)
        forced = list(range(0, n, 5))
        info = {'keyframe_tstamps': np.array(forced)}
        Q = _so3_exp(torch.tensor([[0., 0., .7]])).numpy()[0]
        stub = types.ModuleType('lib.camera.warmstart_droid')
        # pure gauge difference: no change; small disagreement: applied unclamped; diverged rerun: clamped
        for offset, low, high in [(0., 0., 1e-4), (.01, 1e-3, .018), (5.0, .0199, .0201)]:
            def slam(*args, **kwargs):
                self.assertEqual(kwargs['forced_keyframes'], forced)
                self.assertIsNone(kwargs['initial_poses_se3'])
                self.assertFalse(kwargs['use_rendered_for_keyframes'])
                moved = T0.copy()
                moved[20:, 2] += offset  # local disagreement in another gauge (rotation, scale 3, shift)
                return Q @ R0, 3 * moved @ Q.T + np.array([10, 20, 30], dtype=np.float32), info
            stub.run_metric_slam_warmstart = slam
            with patch.dict(sys.modules, {'lib.camera.warmstart_droid': stub}):
                R, T, returned, debug = frontend.pull_back_uniform_keyframes(
                    cam_R=R0, cam_T=T0, forced_keyframes=forced, original_image_dir=str(ROOT), masks=None,
                    intrinsics=None, max_translation_delta=.02, max_rotation_delta_deg=.5)
            self.assertIs(returned, info)
            self.assertTrue(debug['enabled'])
            np.testing.assert_allclose(T[0], T0[0], atol=1e-6)
            change = np.linalg.norm(T.numpy() - T0, axis=-1)
            self.assertLessEqual(change.max(), .02 + 1e-5)
            self.assertTrue(low <= float(change[forced].max()) <= high, (offset, change[forced].max()))
        stub.run_metric_slam_warmstart = lambda *a, **k: (R0, np.full_like(T0, np.nan), info)
        with patch.dict(sys.modules, {'lib.camera.warmstart_droid': stub}):
            R, T, returned, debug = frontend.pull_back_uniform_keyframes(
                cam_R=R0, cam_T=T0, forced_keyframes=forced, original_image_dir=str(ROOT), masks=None,
                intrinsics=None, max_translation_delta=.02, max_rotation_delta_deg=.5)
        self.assertIsNone(returned)
        self.assertIn('uniform_keyframe_pullback_failed', debug['reason'])
        np.testing.assert_array_equal(T, T0)
        R, T, returned, debug = frontend.pull_back_uniform_keyframes(
            cam_R=R0, cam_T=T0, forced_keyframes=[0, 5], original_image_dir=str(ROOT), masks=None, intrinsics=None,
            max_translation_delta=.02, max_rotation_delta_deg=.5)
        self.assertEqual(debug['reason'], 'too_few_uniform_keyframes')
        np.testing.assert_array_equal(T, T0)

    def test_frontdelta_drops_nonfinite_keyframes_instead_of_returning_nan(self):
        import sys
        import types
        n = 8
        old_R = np.repeat(np.eye(3, dtype=np.float32)[None], n, axis=0)
        old_T = np.stack([np.arange(n), np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
        info = {'keyframe_tstamps': list(range(n)), 'keyframe_poses_se3': np.zeros((n, 7)),
                'keyframe_disps': np.ones((n, 1, 1))}
        stub = types.ModuleType('lib.camera.warmstart_droid')
        stub.run_slam_warmstart = lambda *args, **kwargs: (None, info)
        for bad in ([3], [1, 2, 3, 4, 5]):
            new_T = old_T + np.array([0, .01, 0], dtype=np.float32)
            new_T[bad] = np.nan
            with patch.dict(sys.modules, {'lib.camera.warmstart_droid': stub}), patch.object(
                    frontend, '_se3_raw_to_c2w', side_effect=[(old_R, old_T), (old_R.copy(), new_T)]):
                R, T, debug = frontend.refine_camera_with_frontend_delta(
                    cam_R=old_R, cam_T=old_T, keyframe_info=info, render_dir=str(ROOT), original_image_dir=str(ROOT),
                    use_rendered_for_keyframes=False, max_translation_delta=.05, max_rotation_delta_deg=1.)
            self.assertTrue(torch.isfinite(torch.as_tensor(T)).all())
            self.assertEqual(debug['num_nonfinite_keyframes_dropped'], len(bad))
            if len(bad) == 1:
                self.assertTrue(debug['enabled'])
                np.testing.assert_allclose(np.asarray(T)[1:, 1], .01, atol=1e-6)
            else:  # 3 finite keyframes < min_common_keyframes
                self.assertFalse(debug['enabled'])
                self.assertEqual(debug['reason'], 'too_few_finite_frontend_keyframes')
                np.testing.assert_array_equal(T, old_T)

    def test_frontdelta_checks_raw_delta_before_anchor_and_clamp(self):
        import sys
        import types
        old_R = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
        old_T = np.stack([np.arange(4), np.zeros(4), np.zeros(4)], axis=1).astype(np.float32)
        info = {'keyframe_tstamps': [0, 1, 2, 3], 'keyframe_poses_se3': np.zeros((4, 7)),
                'keyframe_disps': np.ones((4, 1, 1))}
        stub = types.ModuleType('lib.camera.warmstart_droid')
        stub.run_slam_warmstart = lambda *args, **kwargs: (None, info)
        for mode in ['large_first_translation', 'rotation180', 'small_delta']:
            new_T, new_R = old_T.copy(), old_R.copy()
            if mode == 'large_first_translation':
                new_T[0, 0] += .1
            elif mode == 'rotation180':
                new_R[2] = np.diag([1., -1., -1.])
            else:
                new_T[:, 1] += .01
            with patch.dict(sys.modules, {'lib.camera.warmstart_droid': stub}), patch.object(
                frontend, '_se3_raw_to_c2w', side_effect=[(old_R, old_T), (new_R, new_T)]
            ):
                R, T, debug = frontend.refine_camera_with_frontend_delta(
                    cam_R=old_R, cam_T=old_T, keyframe_info=info, render_dir=str(ROOT),
                    original_image_dir=str(ROOT), use_rendered_for_keyframes=False,
                    reject_on_excess_delta=True, max_translation_delta=.05, max_rotation_delta_deg=1.)
            if mode == 'small_delta':
                self.assertTrue(debug['accepted'])
                self.assertFalse(debug['fallback_to_baseline'])
                np.testing.assert_array_equal(T[0], old_T[0])
                np.testing.assert_allclose(T[1:, 1], .01)
            else:
                self.assertEqual(debug['reason'], 'requested_delta_exceeds_limits')
                self.assertFalse(debug['accepted'])
                np.testing.assert_array_equal(T, old_T)
                np.testing.assert_array_equal(R, old_R)

    def test_legacy_mode_explicitly_omits_world_rotation(self):
        import sys
        import types
        old_R = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
        old_T = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
        delta_T = np.repeat(np.array([[.02, .01, 0]], dtype=np.float32), 4, axis=0)
        delta_R = _so3_exp(torch.tensor([[.01, 0, 0]] * 4)).numpy()
        Q = _so3_exp(torch.tensor([[0, 0, np.pi / 2]])).numpy()[0]
        base_R = Q @ old_R
        base_T = 2 * (old_T @ Q.T)
        info = {'keyframe_tstamps': [0, 1, 2, 3], 'keyframe_poses_se3': np.zeros((4, 7)),
                'keyframe_disps': np.ones((4, 1, 1))}
        stub = types.ModuleType('lib.camera.warmstart_droid')
        stub.run_slam_warmstart = lambda *args, **kwargs: (None, info)
        with patch.dict(sys.modules, {'lib.camera.warmstart_droid': stub}), patch.object(
            frontend, '_se3_raw_to_c2w', side_effect=[(old_R, old_T), (delta_R @ old_R, old_T + delta_T)]
        ):
            R, T, debug = frontend.refine_camera_with_frontend_delta(
                cam_R=base_R, cam_T=base_T, keyframe_info=info, render_dir=str(ROOT),
                original_image_dir=str(ROOT), use_rendered_for_keyframes=False,
                delta_coordinate_mode='legacy_scale_only', fix_first_keyframe=False,
                max_translation_delta=0, max_rotation_delta_deg=0)
        self.assertEqual(debug['delta_coordinate_frame'], 'legacy_scale_only')
        np.testing.assert_allclose(T, base_T + 2 * delta_T, atol=1e-6)
        np.testing.assert_allclose(R, delta_R @ base_R, atol=1e-6)
        self.assertNotIn('raw_to_world_rotation', debug)

    def test_local_metric_scale_rescales_steps_only(self):
        T0 = np.stack([np.arange(11), np.zeros(11), np.zeros(11)], -1).astype(np.float32)
        info = {'keyframe_tstamps': list(range(0, 11, 2)),
                'keyframe_metric_scales': [2.0, 2.0, 2.0, 2.0, 2.6, 2.6]}  # tail drifts +30%, clipped to +10%
        T, debug = frontend.correct_local_metric_scale(T0, info, median_window=1, max_ratio_deviation=.1,
                                                       strength=1.0, min_keyframes=4)
        self.assertTrue(debug['enabled'])
        ratio = frontend.metric_scale_ratio(info['keyframe_tstamps'], info['keyframe_metric_scales'], 11, 1, .1, 1.0)
        self.assertAlmostEqual(ratio.max(), 1.1)
        np.testing.assert_allclose(T[0], T0[0])
        np.testing.assert_allclose(np.diff(T, axis=0)[:, 0], ratio[1:], atol=1e-6)
        T, debug = frontend.correct_local_metric_scale(T0, dict(info, keyframe_metric_scales=[np.nan] * 6), min_keyframes=4)
        self.assertFalse(debug['enabled'])
        np.testing.assert_allclose(T, T0)
        T, debug = frontend.correct_local_metric_scale(T0, info, enabled=False)
        self.assertEqual(debug['reason'], 'disabled')

    def test_body_model_root_must_be_configured_and_complete(self):
        import lib.gvhmr as gvhmr
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {}, clear=True), \
                patch.object(gvhmr, '_BODY_MODEL_ROOT', None):
            with self.assertRaises(RuntimeError):
                gvhmr.body_model_root()
            gvhmr.set_body_model_root(directory)
            with self.assertRaises(FileNotFoundError):
                gvhmr.body_model_root()
            for name in ('smplx/SMPLX_NEUTRAL.npz', 'smpl/SMPL_NEUTRAL.pkl'):
                (Path(directory) / name).parent.mkdir(parents=True, exist_ok=True)
                (Path(directory) / name).write_bytes(b'')
            self.assertEqual(gvhmr.body_model_root(), Path(directory))
        for name in ('smplx2smpl_sparse.pt', 'smpl_neutral_J_regressor.pt', 'smpl_vert_segmentation.json'):
            self.assertTrue((gvhmr.ASSETS / name).is_file())

    def test_video_head_never_uses_annotation_keypoints_or_rotations(self):
        from lib.pipeline.components.hpe import HPEComponent
        n = 6
        slam_R = _so3_exp(torch.tensor([[0., .1 * i, 0.] for i in range(n)])).numpy()
        gt_ext = np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0)
        K = np.array([[1000., 0, 500], [0, 1000., 400], [0, 0, 1]])
        current = PipelineData(sequence_name='test', image_paths=[f'{i}.jpg' for i in range(n)],
                               bboxes=np.zeros((n, 4), dtype=np.float32),
                               camera_params=CameraParams(R=slam_R, T=np.zeros((n, 3)), intrinsics=K),
                               annotations={'camera': {'intrinsics': K, 'extrinsics': gt_ext},
                                            'kp2d': np.ones((n, 24, 2)), 'vitpose-h': np.ones((n, 25, 3))},
                               metadata={'output_dir': '/tmp'})
        component = HPEComponent('hpe', {'backend': 'prompthmr'})
        calls = {}

        def estimate(**kwargs):
            calls.update(kwargs)
            raise StopIteration
        with patch.object(component.backend, 'estimate_smpl', side_effect=estimate), self.assertRaises(StopIteration):
            component.execute(current)
        self.assertNotIn('kp2d_override', calls)
        np.testing.assert_allclose(calls['cam_R_w2c_override'], slam_R.transpose(0, 2, 1))
        np.testing.assert_allclose(calls['K_fullimg_override'], K)
        self.assertEqual(current.metadata['video_head_camera_source'], 'estimated_slam_not_gt')
        self.assertEqual(current.metadata['keypoint_source'], 'vitpose')


if __name__ == '__main__':
    unittest.main()
