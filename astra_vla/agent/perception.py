"""Evidence-bound multiview 2D perception; no RGB-to-millimetre/contact claims."""
from copy import deepcopy
import math
import time

import cv2
import numpy as np

from .contracts import CAMERAS, ImageEvidence, SceneState, digest, number, unique


class ColorRegions:
    """Executable baseline for configured colour markers, NOT general object recognition.

    Labels are supplied by the experiment's colour/class mapping. Confidence is
    mask solidity, not a calibrated class probability. No inferred physical grasp
    region is emitted. Multiple components remain ambiguous to the selector.
    """
    def __init__(self, profiles, *, minimum_area=20):
        self.profiles = deepcopy(profiles)
        if type(minimum_area) is not int or minimum_area < 3:
            raise ValueError('minimum area must be >=3')
        self.minimum_area = minimum_area
        for label, bounds in self.profiles.items():
            if not isinstance(label, str) or not label or len(bounds) != 2:
                raise ValueError('named HSV bounds required')
            lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
            if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
                    or np.any(lo < 0) or np.any(hi > [179, 255, 255]) or np.any(lo > hi)):
                raise ValueError('invalid HSV range')

    def identity(self):
        return {'provider': 'configured_hsv_components', 'version': '1.0.0',
                'profiles': self.profiles, 'minimum_area': self.minimum_area,
                'opencv': cv2.__version__, 'numpy': np.__version__, 'semantic_model': None}

    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = image.shape[:2]
        results = []
        for label, (lo, hi) in self.profiles.items():
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            count, components, stats, centroids = cv2.connectedComponentsWithStats(mask)
            for i in range(1, count):
                x, y, w, h, area = map(int, stats[i])
                if area < self.minimum_area:
                    continue
                pixels = np.column_stack(np.where(components == i)[::-1]).astype(float)
                values, vectors = np.linalg.eigh(np.cov(pixels.T))
                axis = vectors[:, -1]
                axis = axis if axis[0] >= 0 else -axis
                elongated = values[-1] > 1.5 * max(values[0], 1e-9)
                center = centroids[i]
                projection = (pixels-center) @ axis
                clipped = x == 0 or y == 0 or x+w == width or y+h == height
                results.append(dict(label=label, bbox_xyxy=[x, y, x+w, y+h],
                                    center_px=center.tolist(),
                                    axis_2d=axis.tolist() if elongated and not clipped else None,
                                    keypoints={'axis_start': (center+axis*projection.min()).tolist(),
                                               'axis_end': (center+axis*projection.max()).tolist()}
                                    if elongated and not clipped else {},
                                    actionable_regions=[], confidence=float(area/(w*h)),
                                    confidence_kind='mask_solidity_not_class_probability',
                                    visibility='partial' if clipped else 'visible_region',
                                    occlusion='unknown', area_px=area,
                                    uncertainty=['semantic_label_from_configuration',
                                                 'occlusion_not_observable_from_mask',
                                                 'grasp_region_unknown']))
        return results


def observe_scene(metadata, robot_state, provider, *, mode='offline', now=None, max_age=.6):
    if set(metadata) != set(CAMERAS) or mode not in ('offline', 'live_readonly'):
        raise ValueError('three named source views and explicit mode required')
    now = time.time() if now is None else number(now)
    observation_id = unique('observation')
    views, objects = {}, []
    for camera, row in metadata.items():
        evidence = ImageEvidence.from_metadata(camera, observation_id, row)
        if mode == 'live_readonly' and not 0 <= now-evidence.timestamp <= max_age:
            raise ValueError('stale/future RGB')
        evidence.verify()
        image = cv2.imread(evidence.path)
        if image is None or image.shape[:2] != (evidence.height, evidence.width):
            raise ValueError('image dimensions differ from source metadata')
        views[camera] = evidence
        for index, obj in enumerate(provider.detect(image)):
            obj = deepcopy(obj)
            point = np.asarray(obj['center_px'], dtype=float)
            confidence = number(obj['confidence'])
            if (point.shape != (2,) or not np.isfinite(point).all() or
                    np.any(point < 0) or np.any(point >= [evidence.width, evidence.height]) or
                    not 0 <= confidence <= 1):
                raise ValueError('invalid detector geometry/confidence')
            obj.update(object_id=f'{camera}_{index}', camera_id=camera,
                       observation_id=observation_id, timestamp=evidence.timestamp,
                       evidence_sha256=evidence.sha256)
            objects.append(obj)
        evidence.verify()  # reject replacement during detector execution
    arms = deepcopy(robot_state.get('arms', {}))
    if mode == 'live_readonly':
        if set(arms) != {'left', 'right'}:
            raise ValueError('both SDK feedback streams required')
        for arm in arms.values():
            if not 0 <= now-number(arm['stamp']) <= .3:
                raise ValueError('stale/future SDK feedback')
    scene = SceneState(observation_id, max(v.timestamp for v in views.values()), views,
                       deepcopy(robot_state), {k: v.get('gripper', {}) for k, v in arms.items()},
                       objects=objects, provider_identity=deepcopy(provider.identity()), mode=mode)
    for obj in objects:
        key = obj['object_id']
        scene.object_features[key] = {k: obj[k] for k in ('center_px', 'axis_2d', 'keypoints', 'actionable_regions')}
        scene.visibility[key] = {'visibility': obj['visibility'], 'occlusion': obj['occlusion']}
        scene.confidence[key] = obj['confidence']
    for camera in CAMERAS:
        labels = {o['label'] for o in objects if o['camera_id'] == camera} - {'finger_left', 'finger_right', 'palm'}
        for label in sorted(labels):
            relation = grasp_relation(scene, camera, label)
            scene.spatial_relations.append(dict(camera_id=camera, target=label, **relation))
    return scene


def visual_error(scene, camera, target, desired_px):
    obj = scene.object(camera, target)
    if obj is None or obj['visibility'] != 'visible_region':
        return {'status': 'needs_observation', 'reason': 'target_missing_ambiguous_or_partial'}
    desired = np.asarray(desired_px, dtype=float)
    view = scene.views[camera]
    if (desired.shape != (2,) or not np.isfinite(desired).all() or
            np.any(desired < 0) or np.any(desired >= [view.width, view.height])):
        raise ValueError('desired pixel must lie in the source image')
    error = desired - np.asarray(obj['center_px'])
    return {'status': 'succeeded', 'error_px': error.tolist(), 'norm_px': float(np.linalg.norm(error)),
            'observation_id': scene.observation_id, 'camera_id': camera,
            'timestamp': view.timestamp, 'evidence_sha256': view.sha256}


def grasp_relation(scene, camera, target):
    """Only a 2D projection. Never certifies entry depth, contact or secure grasp."""
    rows = [scene.object(camera, name) for name in (target, 'finger_left', 'finger_right', 'palm')]
    if any(row is None or row['visibility'] != 'visible_region' for row in rows):
        return {'status': 'needs_observation', 'reason': 'gripper_or_target_not_visible'}
    obj, left, right, palm = rows
    a, b, p = (np.array(row['center_px']) for row in (left, right, obj))
    closure = b-a
    width = float(np.linalg.norm(closure))
    approach = (a+b)/2-np.array(palm['center_px'])
    if width < 1 or np.linalg.norm(approach) < 1:
        return {'status': 'needs_observation', 'reason': 'degenerate_gripper_geometry'}
    direction = closure/width
    coordinate = float((p-a) @ direction / width)
    relative = p-(a+b)/2
    return dict(status='succeeded', projected_between_fingers=0 <= coordinate <= 1,
                closure_direction_2d=direction.tolist(), approach_direction_2d=(approach/np.linalg.norm(approach)).tolist(),
                relative_position_finger_widths=[float(relative @ direction/width),
                                                float((direction[0]*relative[1]-direction[1]*relative[0])/width)],
                entry_depth=None, bilateral_contact=None, tip_table_clearance=None,
                object_axis_2d=obj['axis_2d'], observation_id=scene.observation_id,
                evidence_sha256=scene.views[camera].sha256)


def inspect_grasp(before, after, camera, target, *, threshold=.15):
    if number(threshold) <= 0:
        raise ValueError('positive slip threshold required')
    a, b = before.views[camera], after.views[camera]
    if (b.timestamp <= a.timestamp or b.sequence <= a.sequence or a.serial != b.serial or
            (a.width, a.height) != (b.width, b.height) or digest(before.provider_identity) != digest(after.provider_identity)):
        return {'status': 'needs_observation', 'reason': 'incomparable_or_repeated_frames'}
    old, new = grasp_relation(before, camera, target), grasp_relation(after, camera, target)
    if any(r['status'] != 'succeeded' for r in (old, new)):
        return {'status': 'needs_observation', 'reason': 'relative_grasp_unobservable'}
    delta = math.dist(old['relative_position_finger_widths'], new['relative_position_finger_widths'])
    return dict(status='slippage_detected' if delta > threshold else 'succeeded',
                apparent_relative_slip_finger_widths=delta, physical_slippage=None,
                physical_grasp_verified=None, evidence=[a.sha256, b.sha256],
                uncertainty=['perspective_change_or_association_error_can_mimic_slip'])


class LocalVisualServo:
    """Fit a local image Jacobian from measured deltas, never commanded targets.

    Proposal units follow the feedback units supplied by the adapter. Phase one
    only executes these in the isolated image-plane simulator.
    """
    def __init__(self, *, gain=.5, max_step=.1):
        if not 0 < number(gain) <= 1 or number(max_step) <= 0:
            raise ValueError('invalid servo parameters')
        self.gain, self.max_step = gain, max_step
        self.samples = []

    def update(self, actual_delta, image_delta, *, evidence):
        x, y = np.asarray(actual_delta, float), np.asarray(image_delta, float)
        if x.shape != (2,) or y.shape != (2,) or not np.isfinite([x, y]).all() or not evidence:
            raise ValueError('two finite measured deltas and evidence required')
        if np.linalg.norm(x) <= 1e-9:
            raise ValueError('no measured robot displacement')
        self.samples.append((x, y, deepcopy(evidence)))
        self.samples = self.samples[-16:]

    def propose(self, error):
        if len(self.samples) < 2:
            return {'status': 'needs_observation', 'reason': 'local_image_jacobian_unknown'}
        x, y = (np.array([s[i] for s in self.samples]) for i in (0, 1))
        if np.linalg.matrix_rank(x) < 2:
            return {'status': 'needs_observation', 'reason': 'insufficient_feedback_excitation'}
        jacobian = np.linalg.lstsq(x, y, rcond=None)[0].T
        if np.linalg.cond(jacobian) > 100 or np.linalg.norm(x @ jacobian.T-y) > .2*max(np.linalg.norm(y), 1e-9):
            return {'status': 'needs_observation', 'reason': 'unreliable_local_image_jacobian'}
        error = np.asarray(error, float)
        if error.shape != (2,) or not np.isfinite(error).all():
            raise ValueError('invalid visual error')
        delta = self.gain * np.linalg.solve(jacobian, error)
        delta *= min(1., self.max_step/max(float(np.linalg.norm(delta)), 1e-12))
        return {'status': 'succeeded', 'delta': delta.tolist(), 'jacobian': jacobian.tolist(),
                'motion_authorized': False, 'units': 'adapter_feedback_units'}
