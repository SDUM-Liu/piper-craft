"""System 2 proposals and fixed-budget, evidence-checked paired selection.

Offline selection qualifies an offline alias only. No hardware release, generated
code execution, model training or success-map writes are performed here.
"""
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import random
import statistics

from astra_vla.config import atomic_json
from .contracts import digest, file_digest, identifier, locked, number, unique

METRICS = ('task_success', 'completion_time', 'number_of_actions', 'number_of_observations',
           'model_calls', 'recovery_count', 'intervention_count', 'execution_failure',
           'grasp_slippage', 'placement_error')
# A-D are explicitly defined here because the request only spells out E.
ABLATIONS = {
    'A': dict(memory=False, fixed_skills=False, exploration=False, skill_changes=False),
    'B': dict(memory=True, fixed_skills=False, exploration=False, skill_changes=False),
    'C': dict(memory=True, fixed_skills=True, exploration=False, skill_changes=False),
    'D': dict(memory=True, fixed_skills=True, exploration=True, skill_changes=True),
    'E': dict(memory=True, fixed_skills=True, exploration=True, skill_changes=False),
}


def propose_improvement(episodes):
    """Evidence-linked hypotheses; failure localization is not causal proof."""
    if not episodes:
        raise ValueError('episodes required')
    failures, evidence, trajectory_analysis = [], [], []
    for path in map(Path, episodes):
        record = json.loads(path.read_text())
        if record['state'] != 'closed':
            raise ValueError('analyze complete episodes only')
        if record.get('failure_stage'):
            errors = [e['norm_px'] for e in record.get('visual_errors', []) if e.get('norm_px') is not None]
            trajectory_analysis.append(dict(episode_id=record['episode_id'],
                                            error_samples=len(errors), action_count=record.get('action_count'),
                                            error_reduced=errors[-1] < errors[0] if len(errors) >= 2 else None,
                                            unresolved_features=sum(e.get('status') != 'succeeded'
                                                                    for e in record.get('visual_errors', []))))
            failures.append(record['failure_stage'])
            evidence.append({'path': str(path.resolve()), 'sha256': file_digest(path),
                             'episode_id': record['episode_id'], 'reason': record.get('failure_reason')})
    if not failures:
        return {'status': 'inconclusive', 'reason': 'no localized failure evidence'}
    stage = statistics.mode(failures)
    if stage in ('observe_scene', 'detect_object', 'estimate_object_orientation'):
        cause, target = 'perception', 'perception_provider'
    elif (stage == 'align_image' and all(e['reason'] == 'budget_exhausted' for e in evidence)
          and all(t['error_reduced'] is True and t['unresolved_features'] == 0 for t in trajectory_analysis)):
        cause, target = 'skill_or_budget', 'align_image.gain'
    elif stage == 'align_image':
        cause, target = 'execution_or_local_model', 'local_image_jacobian'
    elif stage == 'estimate_visual_error':
        cause, target = 'skill_or_budget', 'align_image.gain'
    else:
        cause, target = 'planning_or_unknown', 'composition'
    return dict(status='hypothesis', problem=f'{len(failures)} episodes with localized failures', evidence=evidence,
                trajectory_analysis=trajectory_analysis,
                failure_stage=stage, cause_hypothesis=cause, modification_target=target,
                expected_improvement='fewer unresolved visual errors within the same action budget',
                validation='freeze one factor; paired held-out scenes; independent outcome evaluator',
                negative_effects=['overshoot', 'loss of visibility', 'regression on previously solved cases'],
                causality='unproven', mutable_factors=1)


class PolicyStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def freeze(self, *, name, parameters, skill_versions, parent=None, hypothesis=None, ablation='D',
               scope='offline_image_alignment'):
        identifier(name)
        if ablation not in ABLATIONS or not skill_versions:
            raise ValueError('ablation and frozen skills required')
        if scope == 'offline_image_alignment':
            target = 'align_image.gain'
            if set(parameters) != {'gain'} or not 0 < number(parameters['gain']) <= 1:
                raise ValueError('image alignment changes only the image-servo gain')
        elif scope == 'offline_view_selection':
            target = 'select_views.first_view'
            if set(parameters) != {'first_view'} or parameters['first_view'] not in ('auxiliary', 'operator'):
                raise ValueError('view selection changes only which wrist is examined first')
        else:
            raise ValueError('unsupported policy scope')
        if parent:
            old = self.load(parent)
            if old['ablation'] != ablation or old['scope'] != scope:
                raise ValueError('candidate must remain in the same ablation group and scope')
            if not ABLATIONS[ablation]['skill_changes'] or not hypothesis or hypothesis.get('status') != 'hypothesis':
                raise ValueError('candidate needs a hypothesis and a skill-changing arm')
            if old['parameters'] == parameters or old['skill_versions'] != skill_versions:
                raise ValueError('candidate must change one parameter, keeping implementation/skills fixed')
            if hypothesis.get('modification_target') != target or not hypothesis.get('evidence'):
                raise ValueError('candidate requires evidence supporting its specific parameter')
            for evidence in hypothesis['evidence']:
                if file_digest(evidence['path']) != evidence['sha256']:
                    raise ValueError('hypothesis evidence changed')
        record = dict(name=name, parameters=deepcopy(parameters), skill_versions=deepcopy(skill_versions),
                      parent=parent, hypothesis=deepcopy(hypothesis), ablation=ablation,
                      scope=scope, sources={str(p.resolve()): file_digest(p)
                                                                for p in sorted(Path(__file__).parent.glob('*.py'))})
        revision = digest(record)
        with locked(self.directory):
            path = self.directory/(revision+'.json')
            if path.exists() and json.loads(path.read_text()) != record:
                raise ValueError('frozen policy changed')
            if not path.exists():
                atomic_json(path, record)
        return revision

    def load(self, revision):
        if not isinstance(revision, str) or len(revision) != 64 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('invalid policy revision')
        row = json.loads((self.directory/(revision+'.json')).read_text())
        if digest(row) != revision:
            raise ValueError('frozen policy changed')
        for path, expected in row['sources'].items():
            if file_digest(path) != expected:
                raise ValueError('frozen implementation changed')
        for evidence in (row.get('hypothesis') or {}).get('evidence', []):
            if file_digest(evidence['path']) != evidence['sha256']:
                raise ValueError('frozen development evidence changed')
        return row

    def restore(self, revision):
        """Load an old version explicitly; does not change the default alias."""
        return self.load(revision)


class PairedHarness:
    def __init__(self, directory, policies):
        self.directory, self.policies = Path(directory), policies

    def preregister(self, *, parent, candidate, development_cases, validation_cases, conditions,
                    minimum_pairs=8, minimum_gain=.05, alpha=.05, comparison_budget=1):
        old, new = self.policies.load(parent), self.policies.load(candidate)
        if new['parent'] != parent or old['scope'] != new['scope']:
            raise ValueError('candidate lineage mismatch')
        if (type(minimum_pairs) is not int or minimum_pairs < 8 or type(comparison_budget) is not int or
                comparison_budget < 1 or not 0 < number(alpha) < 1 or not 0 <= number(minimum_gain) < 1):
            raise ValueError('invalid preregistered statistics')
        required = {'model', 'perception', 'task_distribution', 'control_interface', 'action_budget', 'evaluator'}
        if set(conditions) != required:
            raise ValueError('freeze model/perception/task/control/budget/evaluator conditions')
        if type(conditions['action_budget']) is not int or conditions['action_budget'] <= 0:
            raise ValueError('positive action budget required')
        if any(not isinstance(conditions[k], str) or not conditions[k].strip() for k in required-{'action_budget'}):
            raise ValueError('nonempty frozen identities required')
        dev, val = list(development_cases), list(validation_cases)
        if not dev or not val or len(set(val)) != len(val) or len(set(dev)) != len(dev) or set(dev) & set(val):
            raise ValueError('duplicate or leaked development/validation identities')
        for case in dev+val:
            identifier(case)
        for evidence in new['hypothesis']['evidence']:
            source = json.loads(Path(evidence['path']).read_text())
            if source['scene_id'] not in dev or source['policy_version'] != parent:
                raise ValueError('candidate hypothesis must use parent development episodes only')
        record = dict(comparison_id=unique('comparison'), parent=parent, candidate=candidate,
                      development_cases=dev, validation_cases=val, conditions=deepcopy(conditions),
                      minimum_pairs=minimum_pairs, minimum_gain=minimum_gain, alpha=alpha,
                      comparison_budget=comparison_budget, scope=old['scope'],
                      method='paired_permutation_fixed_n', score='task_success')
        with locked(self.directory):
            # Do not tune on the same declared validation cases in a second round.
            for path in self.directory.glob('comparison_*.json'):
                previous = json.loads(path.read_text())['record']
                if set(previous['validation_cases']) & set(val):
                    raise ValueError('validation cases already claimed by a frozen round')
            atomic_json(self.directory/(record['comparison_id']+'.json'), {'record': record, 'sha256': digest(record)})
        return record['comparison_id']

    def comparison(self, comparison_id):
        identifier(comparison_id)
        row = json.loads((self.directory/(comparison_id+'.json')).read_text())
        if digest(row['record']) != row['sha256']:
            raise ValueError('comparison changed')
        self.policies.load(row['record']['parent'])
        self.policies.load(row['record']['candidate'])
        return row['record']

    def record_trial(self, comparison_id, *, policy, case, episode_path, metrics, evaluator, evidence):
        comparison = self.comparison(comparison_id)
        if policy not in (comparison['parent'], comparison['candidate']) or case not in comparison['validation_cases']:
            raise ValueError('trial outside frozen pair distribution')
        if evaluator != comparison['conditions']['evaluator'] or set(metrics) != set(METRICS):
            raise ValueError('independent evaluator identity/metric schema mismatch')
        for key, value in metrics.items():
            if value is None:
                continue
            if key in ('task_success', 'execution_failure', 'grasp_slippage'):
                if type(value) is not bool:
                    raise ValueError('boolean or unknown metric required')
            elif number(value) < 0:
                raise ValueError('negative metric')
        episode_path = Path(episode_path).resolve()
        episode = json.loads(episode_path.read_text())
        expected_skills = self.policies.load(policy)['skill_versions']
        if (episode['policy_version'] != policy or episode['scene_id'] != case or episode['state'] != 'closed'
                or episode['mode'] != 'offline' or episode['skill_versions'] != expected_skills):
            raise ValueError('episode does not bind this frozen trial')
        if not episode['initial_observation'] or not episode['observation_history']:
            raise ValueError('trial requires observed initial conditions')
        if (episode.get('policy_parameters') != self.policies.load(policy)['parameters'] or
                episode['model_identity'] != comparison['conditions']['model'] or
                episode.get('execution_budget', {}).get('max_actions') != comparison['conditions']['action_budget']):
            raise ValueError('runtime parameters/model/budget differ from frozen protocol')
        environment = episode.get('environment_identity') or {}
        if any(environment.get(k) != comparison['conditions'][k] for k in ('task_distribution', 'control_interface')):
            raise ValueError('environment/control interface differs from frozen protocol')
        for observation in episode['observation_history']:
            if digest(observation['provider_identity']) != comparison['conditions']['perception']:
                raise ValueError('perception changed during trial')
        for metric, field in (('number_of_actions', 'action_count'), ('number_of_observations', 'observation_count'),
                              ('model_calls', 'model_calls'), ('recovery_count', 'recovery_count'),
                              ('intervention_count', 'intervention_count')):
            if metrics[metric] != episode[field]:
                raise ValueError('metrics differ from episode counters')
        receipt = episode.get('evaluation_result')
        if (not receipt or receipt.get('evaluator') != evaluator or
                receipt.get('episode_id') != episode['episode_id'] or
                receipt.get('task_success') is not metrics['task_success']):
            raise ValueError('independent outcome receipt required')
        if not evidence:
            raise ValueError('independent outcome evidence required')
        # All original RGB references must survive unchanged, not only final RGB.
        refs = [{k: row[k] for k in ('path', 'sha256')} for row in episode['evidence_references']]
        refs += deepcopy(evidence)
        refs.append(dict(path=str(episode_path), sha256=file_digest(episode_path)))
        for row in refs:
            if file_digest(row['path']) != row['sha256']:
                raise ValueError('evaluation evidence changed')
        if not any(json.loads(Path(ref['path']).read_text()) == receipt for ref in evidence):
            raise ValueError('outcome receipt must be among independent evidence')
        trial = dict(trial_id=unique('trial'), comparison_id=comparison_id, comparison_sha256=digest(comparison),
                     policy=policy, case=case, episode_id=episode['episode_id'],
                     evaluator=evaluator, metrics=deepcopy(metrics), evidence=refs,
                     initial_state=digest({'environment': environment, 'input': episode['task_input'],
                                           'images': {k: v['sha256'] for k, v in
                                                      (episode['initial_observation'] or {}).get('views', {}).items()}}))
        directory = self.directory/comparison_id
        with locked(directory):
            path = directory/(case+'_'+policy+'.json')
            if path.exists():
                raise ValueError('duplicate trial; failed/unknown trials cannot be overwritten')
            atomic_json(path, dict(record=trial, sha256=digest(trial)))
        return trial

    def select(self, comparison_id):
        comparison = self.comparison(comparison_id)
        directory = self.directory/comparison_id
        report = dict(comparison_id=comparison_id, decision='inconclusive', reason='incomplete_pairs',
                      parent=comparison['parent'], candidate=comparison['candidate'],
                      n=0, mean_difference=None, p_value=None, confidence_interval=None,
                      method=comparison['method'], alpha=comparison['alpha']/comparison['comparison_budget'],
                      evaluation_budget=2*len(comparison['validation_cases']), regressions=[],
                      metrics={}, scope=comparison['scope'])
        pairs = []
        for case in comparison['validation_cases']:
            pair = []
            initial_states = []
            for policy in (comparison['parent'], comparison['candidate']):
                path = directory/(case+'_'+policy+'.json')
                if not path.exists():
                    return report
                envelope = json.loads(path.read_text())
                trial = envelope['record']
                if (digest(trial) != envelope['sha256'] or trial['comparison_sha256'] != digest(comparison) or
                        trial['policy'] != policy or trial['case'] != case):
                    raise ValueError('trial binding changed')
                for ref in trial['evidence']:
                    if file_digest(ref['path']) != ref['sha256']:
                        raise ValueError('trial evidence changed')
                pair.append(trial['metrics'])
                initial_states.append(trial['initial_state'])
            if initial_states[0] != initial_states[1]:
                raise ValueError('paired initial conditions differ')
            pairs.append(pair)
        report['n'] = len(pairs)
        for key in METRICS:
            values = [[row[i][key] for row in pairs] for i in (0, 1)]
            report['metrics'][key] = [statistics.mean(v) if all(x is not None for x in v) else None for v in values]
        if any(row[i]['task_success'] is None for row in pairs for i in (0, 1)):
            report['reason'] = 'unknown_outcomes'
            return report
        differences = [float(b['task_success'])-float(a['task_success']) for a, b in pairs]
        report['mean_difference'] = statistics.mean(differences)
        for case, (a, b) in zip(comparison['validation_cases'], pairs):
            if (a['task_success'] and not b['task_success']) or b['execution_failure'] is True:
                report['regressions'].append(case)
        if len(pairs) < comparison['minimum_pairs']:
            report['reason'] = 'insufficient_pairs'
        elif report['regressions']:
            report.update(decision='retain_parent', reason='observed_regressions')
        elif any(row[i]['execution_failure'] is None for row in pairs for i in (0, 1)):
            report['reason'] = 'unknown_execution_failures'
        else:
            # Zero pairs contribute neither sign nor evidence. Exact for <=16 discordant pairs.
            nonzero = [abs(d) for d in differences if d]
            observed = sum(differences)
            if not nonzero:
                p = 1.
            elif len(nonzero) <= 16:
                totals = [sum(a*b for a, b in zip(signs, nonzero))
                          for signs in itertools.product((-1, 1), repeat=len(nonzero))]
                p = sum(t >= observed-1e-12 for t in totals)/len(totals)
            else:
                rng = random.Random(0)
                count = sum(sum(rng.choice((-1, 1))*d for d in nonzero) >= observed-1e-12 for _ in range(20000))
                p = (count+1)/20001  # add-one finite Monte Carlo test
            # Conservative bounded paired Hoeffding CI, not a normal approximation.
            radius = math.sqrt(2*math.log(2/report['alpha'])/len(pairs))
            mean = report['mean_difference']
            report.update(p_value=p, confidence_interval=[max(-1., mean-radius), min(1., mean+radius)],
                          confidence_interval_method='paired_hoeffding_two_sided')
            passed = p <= report['alpha'] and mean >= comparison['minimum_gain'] and mean > 0
            report.update(decision='inherit' if passed else 'retain_parent',
                          reason='paired_evidence' if passed else 'no_demonstrated_improvement')
        with locked(directory):
            atomic_json(directory/'selection.json', report)
        return report

    def promote_offline(self, comparison_id):
        report = self.select(comparison_id)  # recompute and recheck every evidence reference
        if report['decision'] != 'inherit':
            raise ValueError('candidate not validated')
        with locked(self.policies.directory):
            path = self.policies.directory/'offline_default.json'
            if path.exists() and json.loads(path.read_text())['revision'] not in (report['parent'], report['candidate']):
                raise ValueError('default parent changed since comparison')
            atomic_json(path, {'revision': report['candidate'], 'parent': report['parent'],
                               'comparison_id': comparison_id, 'selection_sha256': digest(report),
                               'scope': report['scope']})
        return report
