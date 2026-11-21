"""Piper Craft local Agent, scene inspection and isolated validation."""
import argparse
import json
from pathlib import Path

from astra_vla.config import atomic_json
from .experiment import run_experiment
from .perception import ColorRegions, observe_scene
from .runtime import ReadOnlyPiper


def add_session_commands(sub):
    auth = sub.add_parser('auth', help='Piper Craft Sign in with ChatGPT accounts')
    commands = auth.add_subparsers(dest='account_command', required=True)
    for name in ('login', 'list', 'select', 'logout', 'models'):
        command = commands.add_parser(name)
        if name != 'list':
            command.add_argument('--account', required=name == 'select')
    session = sub.add_parser('session', help='run or resume the exact active task (live read-only)')
    session.add_argument('--task-id', required=True)
    session.add_argument('--contract', type=Path, required=True)
    session.add_argument('--account')
    session.add_argument('--model', required=True)
    session.add_argument('--skill-directory', type=Path, help='validated offline viewing-order policy')
    session.add_argument('--skill-revision', help='explicit revision, including the parent for rollback')
    session.add_argument('--max-turns', type=int, default=32, choices=range(1, 101), metavar='1..100')
    show = sub.add_parser('state', help='read the persisted snapshot by task ID')
    show.add_argument('--task-id', required=True)
    sim = sub.add_parser('simulate', help='isolated shuttlecock loop and interruption benchmark')
    sim.add_argument('--output', type=Path, required=True)
    sim.add_argument('--model', help='optional real ChatGPT model; sends only synthetic observations')
    sim.add_argument('--account')
    views = sub.add_parser('validate-views', help='freeze and independently evaluate a view skill candidate')
    views.add_argument('--output', type=Path, required=True)
    views.add_argument('--proposal', type=Path, help='candidate proposal object from an Episode')
    views.add_argument('--regression', action='store_true', help='include a held-out regression case')


def authenticated_model(auth, account, model_name):
    from .responses import Responses

    account = auth.active_label(account)
    if model_name not in {row['slug'] for row in auth.models(account)}:
        raise ValueError('selected model is not listed for this ChatGPT account')
    return Responses(auth, account, model_name)


def run_session_command(args):
    from astra_vla.task_archive import TaskArchive
    from .chatgpt import ChatGPT
    if args.command == 'validate-views':
        from .view_validation import validate_views
        return validate_views(args.output, proposal=args.proposal, regression=args.regression)
    if args.command == 'simulate':
        from .shuttle_sim import run_fixture
        model = authenticated_model(ChatGPT(), args.account, args.model) if args.model else None
        return run_fixture(args.output, remote_model=model)
    archive = TaskArchive()
    if args.command == 'state':
        return json.loads((archive.path(args.task_id)/'agent_state.json').read_text())
    auth = ChatGPT()
    if args.command == 'auth':
        command = args.account_command
        if command == 'list':
            return auth.store.list()
        if command == 'login':
            return auth.sign_in(args.account)
        if command == 'logout':
            return auth.sign_out(args.account)
        if command == 'select':
            auth.store.select(args.account)
            return {'account': args.account, 'active': True}
        return auth.models(auth.active_label(args.account))
    from .session import LiveBackend, task_session
    model = authenticated_model(auth, args.account, args.model)
    contract = json.loads(args.contract.read_text())
    if args.skill_revision and not args.skill_directory:
        raise ValueError('--skill-revision requires --skill-directory')
    skills = None
    if args.skill_directory:
        from .view_validation import pinned_skills
        skills = pinned_skills(args.skill_directory, args.skill_revision)
    with task_session(archive, args.task_id, model, contract, LiveBackend(), skills=skills) as session:
        return session.run(max_turns=args.max_turns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    add_session_commands(sub)
    inspect = sub.add_parser('inspect', help='RGB/SDK read only; no motion or control-session entry')
    inspect.add_argument('--profiles', type=Path, required=True, help='JSON label -> [HSV lower, HSV upper]')
    inspect.add_argument('--observation', type=Path, help='existing observation.json for offline inspection')
    inspect.add_argument('--feedback', type=Path, help='existing SDK status JSON (offline inspection only)')
    inspect.add_argument('--output', type=Path, required=True, help='new directory, never overwrite source evidence')
    experiment = sub.add_parser('experiment')
    experiment.add_argument('--output', type=Path, required=True)
    experiment.add_argument('--pairs', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'experiment':
        result = run_experiment(args.output, pairs=args.pairs)
    elif args.command == 'inspect':
        if args.feedback and not args.observation:
            parser.error('--feedback requires --observation; live SDK is read directly')
        args.output.mkdir(parents=True, exist_ok=False)
        provider = ColorRegions(json.loads(args.profiles.read_text()))
        if args.observation:
            metadata = json.loads(args.observation.read_text())
            feedback = json.loads(args.feedback.read_text()) if args.feedback else {}
            scene = observe_scene(metadata, feedback, provider)
        else:
            scene = ReadOnlyPiper(provider).observe(args.output/'source')
        atomic_json(args.output/'scene.json', scene.to_dict())
        result = {'scene': str((args.output/'scene.json').resolve()), 'observation_id': scene.observation_id,
                  'mode': scene.mode, 'object_count': len(scene.objects), 'motion_sent': False}
    else:
        try:
            result = run_session_command(args)
        except (ValueError, RuntimeError, OSError) as exc:
            parser.exit(1, str(exc)+'\n')
        except KeyboardInterrupt:
            parser.exit(130, 'Cancelled.\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == 'session' and result['status'] != 'completed':
        raise SystemExit(130 if result['status'] == 'cancelled' else 1)
    if args.command == 'simulate' and any(run['execution_status'] != 'completed' or
                                         not run['independent_evaluation']['task_success']
                                         for run in result.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
