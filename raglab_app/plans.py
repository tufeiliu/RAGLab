"""Turn New-experiment form choices into an explicit raglab.experiments plan."""
from __future__ import annotations

from itertools import product

from raglab_app.options import TECHNIQUES

MAX_RUNS = 32


def parse_ints(text: str, *, label: str, low: int, high: int) -> list[int]:
    """'300, 500' -> [300, 500]; rejects anything that is not a whole number in range."""
    values = []
    for part in str(text).replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        if not part.lstrip('-').isdigit():
            raise ValueError(f'{label}: {part!r} is not a whole number')
        value = int(part)
        if not low <= value <= high:
            raise ValueError(f'{label}: {value} is outside {low}–{high}')
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f'{label}: give at least one value')
    return values


def build_plan(*, chat: str, judge: str | None, embeddings: list[str], chunk_sizes: list[int],
               chunk_overlap: int, top_ks: list[int], techniques: list[str], combine: bool,
               baseline: bool) -> dict:
    """Every combination of embedding x chunk size x top-K, crossed with the technique variants.

    combine=False tests each technique on its own; combine=True tests them all together.
    """
    if not embeddings:
        raise ValueError('Choose at least one embedding model')
    if any(chunk_overlap >= size for size in chunk_sizes):
        raise ValueError(f'Chunk overlap ({chunk_overlap}) must be smaller than every chunk size')
    unknown = set(techniques) - set(TECHNIQUES)
    if unknown:
        raise ValueError(f'Unknown technique(s): {sorted(unknown)}')
    variants = []
    if baseline or not techniques:
        variants.append(('baseline', {}))
    if techniques and combine:
        merged = {}
        for key in techniques:
            merged.update(TECHNIQUES[key][2])
        variants.append(('+'.join(techniques), merged))
    elif techniques:
        variants += [(key, dict(TECHNIQUES[key][2])) for key in techniques]
    grid = list(product(embeddings, chunk_sizes, top_ks))
    if len(grid) * len(variants) > MAX_RUNS:
        raise ValueError(f'{len(grid) * len(variants)} configurations exceed the limit of {MAX_RUNS}. '
                         'Choose fewer values; every configuration answers every test question.')
    runs = []
    for (embedding, size, k), (variant, overrides) in product(grid, variants):
        parts = [variant]
        if len(embeddings) > 1:
            parts.append(embedding)
        if len(chunk_sizes) > 1:
            parts.append(f'chunk{size}')
        if len(top_ks) > 1:
            parts.append(f'k{k}')
        runs.append({'name': ' · '.join(parts), 'techniques': overrides,
                     'settings': {'embedding': embedding, 'chunk_size': size,
                                  'chunk_overlap': chunk_overlap, 'top_k': k}})
    return {'settings': {'chat': chat, 'judge': judge or chat, 'judge_enabled': judge is not None},
            'runs': runs}
