"""Selective read-only retrieval from existing Memory; never promotes RGB guesses."""
from copy import deepcopy
import json
from pathlib import Path
import re

from astra_vla.task_memory import TaskMemory
from .contracts import digest


def tokens(text):
    # English words and Chinese bigrams avoid requiring a model/tokenizer.
    result = set(re.findall(r'[a-z0-9_]+', text.lower()))
    for word in re.findall(r'[\u4e00-\u9fff]+', text):
        result.update(word[i:i+2] for i in range(max(1, len(word)-1)))
    return result


def retrieve(root, *, task, objects=(), scene='', start_state='', skills=(), failures=(), limit=4):
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('bounded retrieval limit required')
    directory = Path(root)/'docs/memory'
    state_path = directory/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    maps = TaskMemory(directory, root=root).read()['maps']
    query = tokens(' '.join([task, *objects, scene, start_state, *skills, *failures]))
    candidates = []
    for kind, rows in (('semantic', maps), ('procedural', state.get('problems', []))):
        for row in rows:
            score = len(query & tokens(json.dumps(row, ensure_ascii=False)))
            if score:
                candidates.append((score, kind, row))
    candidates.sort(key=lambda item: -item[0])
    entries = []
    for score, kind, row in candidates[:limit]:
        if kind == 'semantic':
            # Keep applicability and limits with summaries; do not dump all graphs/demos.
            content = {k: deepcopy(row.get(k)) for k in
                       ('id', 'revision', 'task', 'scene_assumptions', 'non_transferable', 'success_criteria')}
            content['facts'] = [case.get('summary', '') for case in row.get('cases', [])]
        else:
            content = {k: deepcopy(row.get(k)) for k in ('topic', 'observed', 'inference', 'change', 'validation')}
        entries.append({'kind': kind, 'score': score, 'content': content, 'applicability': 'assessment_required'})
    return {'revision': digest({'maps': maps, 'problems': state.get('problems', [])}),
            'entries': entries, 'reference_only': True, 'demonstrations_included': False}
