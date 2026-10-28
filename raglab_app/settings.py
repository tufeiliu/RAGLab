"""App defaults from settings.yaml, checked against models.yaml and the known techniques.

Anything missing or invalid falls back to the built-in default, and is reported in
`problems` so the app can say what was ignored instead of failing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from raglab_app import ROOT
from raglab_app.options import TECHNIQUES

SETTINGS_FILE = ROOT / 'settings.yaml'
BUILT_IN = {'answer_model': None, 'embedding_model': None, 'chunk_size': 800, 'chunk_overlap': 150, 'top_k': 5,
            'techniques': [], 'judge': True, 'reuse_cached_answers': False}
BUILT_IN_LIMITS = {'chunk_size': (50, 8000), 'top_k': (1, 50)}


@dataclass
class Settings:
    answer_model: str | None
    embedding_model: str | None
    chunk_size: int
    chunk_overlap: int
    top_k: int
    techniques: list[str]
    judge: bool
    reuse_cached_answers: bool
    limits: dict[str, tuple[int, int]]
    problems: list[str] = field(default_factory=list)


def load(chat_models: dict, embedding_models: dict) -> Settings:
    """chat_models / embedding_models: alias -> label from models.yaml."""
    problems = []
    try:
        raw = yaml.safe_load(SETTINGS_FILE.read_text(encoding='utf-8')) if SETTINGS_FILE.is_file() else {}
    except yaml.YAMLError as exc:
        problems.append(f'settings.yaml could not be read ({exc}); using built-in defaults')
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    given = raw.get('defaults') or {}
    values = {**BUILT_IN, **{k: v for k, v in given.items() if k in BUILT_IN}}
    problems += [f'unknown setting "{k}" ignored' for k in given if k not in BUILT_IN]

    limits = dict(BUILT_IN_LIMITS)
    for name, pair in (raw.get('limits') or {}).items():
        if name in limits and isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, int) for v in pair) \
                and 0 < pair[0] < pair[1]:
            limits[name] = tuple(pair)
        else:
            problems.append(f'limits.{name} must be [smallest, largest] positive whole numbers; ignored')

    for key, models in (('answer_model', chat_models), ('embedding_model', embedding_models)):
        if values[key] not in models:
            if values[key] is not None:
                problems.append(f'{key} "{values[key]}" is not in models.yaml; using {next(iter(models))}')
            values[key] = next(iter(models))
    for key in ('chunk_size', 'top_k'):
        low, high = limits[key]
        if not isinstance(values[key], int) or not low <= values[key] <= high:
            problems.append(f'{key} must be a whole number from {low} to {high}; using {BUILT_IN[key]}')
            values[key] = min(max(BUILT_IN[key], low), high)
    if not isinstance(values['chunk_overlap'], int) or not 0 <= values['chunk_overlap'] < values['chunk_size']:
        fallback = min(BUILT_IN['chunk_overlap'], values['chunk_size'] // 4)
        problems.append(f'chunk_overlap must be from 0 to below chunk_size; using {fallback}')
        values['chunk_overlap'] = fallback
    techniques = values['techniques'] if isinstance(values['techniques'], list) else []
    unknown = [t for t in techniques if t not in TECHNIQUES]
    if unknown:
        problems.append(f'unknown techniques ignored: {", ".join(map(str, unknown))}')
    values['techniques'] = [t for t in techniques if t in TECHNIQUES]
    for key in ('judge', 'reuse_cached_answers'):
        if not isinstance(values[key], bool):
            problems.append(f'{key} must be true or false; using {str(BUILT_IN[key]).lower()}')
            values[key] = BUILT_IN[key]
    return Settings(**values, limits=limits, problems=problems)
