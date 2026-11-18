"""Reproducible CPU image-plane experiment. This is NOT a Piper dynamics simulator."""
import argparse
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from astra_vla.config import ROOT, atomic_json
from astra_vla.task_archive import TaskArchive
from .contracts import digest, file_digest
from .harness import ABLATIONS, METRICS, PairedHarness, PolicyStore, propose_improvement
from .perception import ColorRegions, observe_scene
from .runtime import Agent, register_builtins
from .skills import SkillRegistry

PROFILES = {'target': [[0, 180, 180], [8, 255, 255]]}


class ImagePlaneSimulator:
    """Renderer plus measured abstract XY feedback, isolated from robot imports.

    Units are abstract coordinates, not metres. Warmup samples are measurements
    from an isolated simulator calibration, never claimed as real SDK evidence.
    """
    mode = 'offline'

    def __init__(self, seed):
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.center = rng.uniform([28, 30], [65, 65])
        self.matrix = np.array([[100., 8.], [-5., 90.]])
        self.sequence = 0
        self.provider = ColorRegions(PROFILES)

    def initial_measurements(self):
        samples = []
        for delta in ([.1, 0], [0, .1]):
            measured = np.array(delta)
            samples.append(dict(actual_delta=measured.tolist(), image_delta=(self.matrix @ measured).tolist(),
                                evidence=['simulated_calibration_'+digest(delta)]))
        return samples

    def identity(self):
        return {'task_distribution': 'image-plane-uniform-center-v1', 'control_interface': 'simulated_xy/v1',
                'reset_seed': self.seed, 'units': 'abstract', 'numpy': np.__version__}

    def observe(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.sequence += 1
        rows = {}
        for camera in ('top', 'left_wrist', 'right_wrist'):
            image = np.zeros((128, 160, 3), np.uint8)
            cv2.ellipse(image, tuple(np.round(self.center).astype(int)), (8, 4), 25, 0, 360, (0, 0, 255), -1)
            path = directory/(camera+'.png')
            if not cv2.imwrite(str(path), image):
                raise RuntimeError('failed to write simulator source')
            rows[camera] = dict(available=True, lossless=True, resized=False, width=160, height=128,
                                stamp=1000.+self.sequence*.1, sequence=self.sequence,
                                serial='simulated_'+camera, path=str(path.resolve()))
        atomic_json(directory/'observation.json', rows)
        return observe_scene(rows, {'source': 'simulator', 'arms': {}}, self.provider)

    def move(self, delta):
        actual = np.asarray(delta, dtype=float)
        if actual.shape != (2,) or not np.isfinite(actual).all() or np.linalg.norm(actual) > .25000001:
            raise ValueError('invalid simulator action')
        self.center += self.matrix @ actual
        return dict(actual_delta=actual.tolist(), source='simulator_measured_xy', units='abstract')

    def evaluate(self, episode_path, destination):
        """Independent renderer state, not Agent status, detector or SDK arrival."""
        episode_path = Path(episode_path)
        episode = json.loads(episode_path.read_text())
        distance = float(np.linalg.norm(self.center-np.asarray(destination)))
        success = distance <= 2.
        receipt = dict(evaluator='image_plane_ground_truth/v1', seed=self.seed, final_center=self.center.tolist(),
                       destination=destination, pixel_error=distance, task_success=success,
                       physical_success=None, mode='offline', episode_id=episode['episode_id'])
        path = episode_path.parent/'evaluation.json'
        atomic_json(path, receipt)
        # Store independent result separately from model judgment/execution result.
        episode.update(evaluation_result=receipt, task_outcome='success' if success else 'failure')
        atomic_json(episode_path, episode)
        metrics = dict.fromkeys(METRICS)
        metrics.update(task_success=success, completion_time=episode['execution_time'] if success else None,
                       number_of_actions=episode['action_count'], number_of_observations=episode['observation_count'],
                       model_calls=episode['model_calls'], recovery_count=episode['recovery_count'],
                       intervention_count=episode['intervention_count'],
                       execution_failure=episode['execution_status'] == 'error')
        # Pixel alignment error is in the receipt, not mislabeled as 3D placement error.
        return metrics, [{'path': str(path.resolve()), 'sha256': file_digest(path)}]


def run_experiment(output, *, pairs=8):
    if type(pairs) is not int or not 1 <= pairs <= 32:
        raise ValueError('pair budget must be between 1 and 32')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('use an empty output directory; frozen evidence is never overwritten')
    registry = SkillRegistry(output/'skills')
    skills = register_builtins(registry)
    policies = PolicyStore(output/'policies')
    parent = policies.freeze(name='parent', parameters={'gain': .18}, skill_versions=skills)
    archive = TaskArchive(output)
    archive.begin('离线二维图像对齐实验', 'paired frozen policies; no hardware or physical success claims')
    destination = [110., 90.]

    def trial(seed, case, revision):
        policy = policies.load(revision)
        simulator = ImagePlaneSimulator(seed)
        agent = Agent(registry, archive, simulator, memory_root=output, skills=policy['skill_versions'])
        result = agent.execute_task(task='align_image', target_object='target', destination=destination,
                                    scene_id=case, gain=policy['parameters']['gain'], max_actions=8,
                                    policy_version=revision)
        metrics, evidence = simulator.evaluate(result['episode_path'], destination)
        return result, metrics, evidence

    development = []
    for seed in (1, 2):
        result, _, _ = trial(seed, f'development_{seed}', parent)
        development.append(result['episode_path'])
    hypothesis = propose_improvement(development)
    atomic_json(output/'hypothesis.json', hypothesis)
    candidate = policies.freeze(name='candidate', parameters={'gain': .65}, skill_versions=skills,
                                parent=parent, hypothesis=hypothesis)
    harness = PairedHarness(output/'experiments', policies)
    conditions = dict(model='none/local-rules', perception=digest(ImagePlaneSimulator(1).provider.identity()),
                      task_distribution='image-plane-uniform-center-v1', control_interface='simulated_xy/v1',
                      action_budget=8, evaluator='image_plane_ground_truth/v1')
    comparison_id = harness.preregister(parent=parent, candidate=candidate,
                                        development_cases=['development_1', 'development_2'],
                                        validation_cases=[f'validation_{i}' for i in range(101, 101+pairs)],
                                        conditions=conditions)
    for index, seed in enumerate(range(101, 101+pairs)):
        # Alternate order to avoid systematic parent-first timing bias.
        order = (parent, candidate) if index % 2 == 0 else (candidate, parent)
        for revision in order:
            case = f'validation_{seed}'
            result, metrics, evidence = trial(seed, case, revision)
            harness.record_trial(comparison_id, policy=revision, case=case, episode_path=result['episode_path'],
                                 metrics=metrics, evaluator=conditions['evaluator'], evidence=evidence)
    report = harness.select(comparison_id)
    if report['decision'] == 'inherit':
        harness.promote_offline(comparison_id)
    report['limitations'] = ['synthetic 2D colour target; not real object recognition or Piper dynamics',
                             'gain-only candidate; no implementation mutation tested',
                             'no physical success, grasp or speed improvement claimed',
                             'Memory and model effects require a separate real task distribution']
    atomic_json(output/'report.json', report)
    atomic_json(output/'ablation_protocol.json', {'groups': ABLATIONS, 'conditions': conditions,
                                                'status': 'preregister_template_not_executed',
                                                'unknown_questions': ['Memory benefit', 'reasoning cost benefit',
                                                                      'exploration vs implementation changes']})
    # Unknown physical outcome avoids inserting synthetic successes in task maps.
    archive.finish(outcome='unknown', summary='离线二维配对实验完成；实体效果未知；原始实验材料保留供回归。')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pairs', type=int, default=8)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix='piper-agent-'))
    if output.resolve() == ROOT or output.resolve() == ROOT/'runs':
        parser.error('use a separate experiment workspace')
    report = run_experiment(output, pairs=args.pairs)
    print(json.dumps({'output': str(output), **report}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
