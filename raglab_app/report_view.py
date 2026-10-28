"""Results of a finished evaluation: scorecard, per-category charts, setup comparison, question-level diagnosis.

Works for general evaluations (raglab.experiments) and FinanceBench reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from raglab_app.response_explorer import pair_question_rows, row_view
from raglab_app.results_charts import (ANSWER_METRICS, EVIDENCE_KEYS, JUDGED_KEYS, LEGACY_EVIDENCE_KEYS, RETRIEVAL_METRICS,
                                       available, by_category, category_chart, display, rating, score_chart)

FLAG_LABELS = {
    'retrieval_keyword_gap': 'Expected keywords not retrieved',
    'answer_accuracy_failure': 'Wrong answer',
    'grounding_failure': 'Answer not supported by passages',
    'possible_generation_failure': 'Right passages, wrong answer',
    'no_retrieved_documents': 'Nothing retrieved',
}
SETTING_LABELS = {
    'chunk_size': 'Chunk size', 'chunk_overlap': 'Overlap', 'top_k': 'Top-K',
    'retrieval': 'Retrieval', 'window_lines': 'Window (lines)', 'window_overlap': 'Window overlap',
}
TECHNIQUE_KEYS = ('query_rewrite', 'query_fusion', 'query_expansion', 'rerank', 'improved_prompt',
                  'preprocess', 'hierarchical', 'graph', 'agentic', 'opening')


@st.cache_data(max_entries=8, show_spinner='Loading results…')
def load(job_id: str, report_path: str, kind: str, modified: float) -> dict:
    """modified (the file's mtime) is part of the cache key, so a rewritten report reloads."""
    if kind == 'financial':
        from ui_backend import load_report  # Repairs evidence metrics in older reports.
        return add_evidence_scores(load_report(report_path))
    return json.loads(Path(report_path).read_text(encoding='utf-8'))


def add_evidence_scores(report: dict) -> dict:
    """FinanceBench reports from before general evidence scoring: score the stored passages now (no API calls)."""
    from statistics import mean
    from raglab.evidence import evidence_metrics
    from raglab_app.benchmarks import FINANCE_DATA
    experiments = report.get('experiments', [])
    gold_file = FINANCE_DATA / 'questions.jsonl'
    if not experiments or 'evidence_recall' in experiments[0].get('summary', {}) or not gold_file.is_file():
        return report
    gold = {g['question']: g.get('evidence') or [] for g in
            (json.loads(line) for line in gold_file.read_text(encoding='utf-8').splitlines() if line.strip())}
    for experiment in experiments:
        k = int(experiment.get('config', {}).get('top_k', 5))
        for row in experiment.get('results', []):
            row.update(evidence_metrics(row.get('retrieved_documents', []), gold.get(row.get('question')), k=k))
        for key in EVIDENCE_KEYS:
            values = [r[key] for r in experiment.get('results', []) if r.get(key) is not None]
            experiment['summary'][key] = mean(values) if values else None
    return report


def fmt(value, digits=2):
    return None if value is None else round(float(value), digits)


def retrieval_metrics_for(summaries: list[dict]) -> dict:
    """The retrieval metrics worth showing for these runs, most precise first."""
    metrics = available(summaries, RETRIEVAL_METRICS)
    finance_legacy = 'ndcg_at_k' in metrics  # old FinanceBench reports filled mrr/ndcg from evidence
    if any(k in metrics for k in EVIDENCE_KEYS):
        # Gold evidence measures the same thing as the judge's view, more objectively: show one set.
        for key in JUDGED_KEYS:
            metrics.pop(key, None)
        for key in LEGACY_EVIDENCE_KEYS + (('mrr', 'ndcg') if finance_legacy else ()):
            metrics.pop(key, None)
    elif finance_legacy:
        metrics.pop('ndcg', None)
        if 'mrr' in metrics:
            metrics['mrr'] = ('MRR', 'How near the top the first passage overlapping the gold evidence is (0–1).')
    return metrics


def scorecard(values: dict, metrics: dict) -> None:
    """One tile per measured metric, with a Good/Fair/Poor badge. Hover the ? for what it means."""
    with st.container(horizontal=True):
        for key, (label, description) in metrics.items():
            value = values.get(key)
            if value is None:
                continue
            with st.container(border=True):
                st.metric(label, display(key, float(value)), help=description)
                verdict, colour = rating(key, float(value))
                st.badge(verdict, color=colour)


def score_badges(values: dict, metrics: dict) -> str:
    """Compact one-line scores as coloured badges (markdown), for a single answer."""
    badges = []
    for key, (label, _) in metrics.items():
        value = values.get(key)
        if value is not None:
            _, colour = rating(key, float(value))
            badges.append(f':{colour}-badge[{label} {display(key, float(value))}]')
    return ' '.join(badges)


def techniques(config: dict) -> str:
    on = [k.replace('_', ' ') + (f' ×{config[k]}' if k == 'query_expansion' else '')
          for k in TECHNIQUE_KEYS if config.get(k) and not (k == 'query_expansion' and config[k] == 1)]
    return ', '.join(on) or 'none'


def summary(experiment: dict, report: dict, *, charts: bool = False) -> None:
    """Scorecard (plus bar charts when charts=True) and per-category breakdown for one setup."""
    s = experiment.get('summary', {})
    retrieval = retrieval_metrics_for([s])
    answer = available([s], ANSWER_METRICS)
    st.markdown('**Retrieval: did the search find the right passages?**')
    if retrieval:
        scorecard(s, retrieval)
    else:
        st.caption('Not measured: give the test questions correct answers (with the judge on), passages or keywords.')
    st.markdown('**Answers: are they correct, complete and grounded?**')
    if answer:
        scorecard(s, answer)
    else:
        st.caption('Not measured: the LLM judge was off, or the test questions have no reference answers.')
    st.caption(f"{s.get('successful', 0)} of {s.get('total', 0)} questions answered"
               + (f" · {s['latency_s']:.1f}s per question on average" if s.get('latency_s') else '')
               + ' · Good / Fair thresholds: 0.9 / 0.75 for retrieval, 4.5 / 4.0 for answers.')
    if charts:
        score_charts([{'name': experiment['name'], **s}])

    rows = experiment.get('results', [])
    main_retrieval = next((k for k in ('evidence_recall', 'context_recall', 'mrr') if s.get(k) is not None), 'mrr')
    charts = [(title, frame, top) for title, frame, top in (
        (f'Average {RETRIEVAL_METRICS[main_retrieval][0].lower()}', by_category(rows, main_retrieval), 1.0),
        ('Average accuracy (1–5)', by_category(rows, 'accuracy'), 5.0)) if len(frame) > 1]
    if charts:
        st.markdown('**By question category:** which kinds of question are weak?')
        for column, (title, frame, top) in zip(st.columns(len(charts)), charts):
            with column:
                st.altair_chart(category_chart(frame, title=title, max_score=top), alt=f'{title} by category')


def comparison(report: dict) -> None:
    """Table and charts comparing two or more setups."""
    experiments = report.get('experiments', [])
    summaries = [{'name': e['name'], **e.get('summary', {})} for e in experiments]
    answer = available(summaries, ANSWER_METRICS)
    retrieval = retrieval_metrics_for(summaries)
    st.subheader('Which setup did best?')
    st.caption('Each row is one setup, run on the same questions. Higher is better for every score. '
               'Hover a column name for what it measures.')
    table = []
    for e, s in zip(experiments, summaries):
        row = {'Setup': e['name']}
        row.update({label: fmt(s.get(k)) for k, (label, _) in answer.items()})
        row.update({label: fmt(s.get(k), 3) for k, (label, _) in retrieval.items()})
        row['Seconds per question'] = fmt(s.get('latency_s'), 1)
        table.append(row)
    help_text = {label: text for label, text in [*answer.values(), *retrieval.values()]}
    st.dataframe(pd.DataFrame(table).dropna(axis=1, how='all'), hide_index=True,
                 column_config={c: st.column_config.Column(help=h) for c, h in help_text.items()})
    missing = []
    if not retrieval:
        missing.append('**search** scores: add correct answers (judged), the passage that contains the answer, '
                       'or keywords to the test questions')
    if not answer:
        missing.append('**answer** scores: add correct answers to the test questions and keep the LLM judge on')
    for item in missing:
        st.caption(f':material/info: Not measured: {item}.')
    st.markdown('**At a glance:** green good, yellow fair, red poor')
    st.dataframe(rating_table(summaries, {**answer, **retrieval}))
    score_charts(summaries)
    settings_table(experiments)


def score_charts(summaries: list[dict]) -> None:
    """Bar charts of answer and retrieval scores, one bar per setup."""
    charts = [(label, score_chart(summaries, metrics, max_score=top))
              for label, metrics, top in (('Answer quality', available(summaries, ANSWER_METRICS), 5.0),
                                          ('Retrieval quality', retrieval_metrics_for(summaries), 1.0))
              if metrics]
    for column, (label, chart) in zip(st.columns(len(charts)) if charts else [], charts):
        with column:
            st.markdown(f'**{label}**')
            st.altair_chart(chart, alt=f'{label} by setup')


def settings_table(experiments: list[dict]) -> None:
    with st.expander('What differs between the setups' if len(experiments) > 1 else 'Settings used'):
        rows = []
        for e in experiments:
            config, models = e.get('config', {}), e.get('models', {})
            row = {'Setup': e['name'], 'Answer model': models.get('chat'),
                   'Embedding model': models.get('embedding'), 'Judge': models.get('judge') or 'off'}
            row.update({label: config.get(k) for k, label in SETTING_LABELS.items() if k in config})
            row['Techniques'] = techniques(config)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def show_passages(passages: list[dict]) -> None:
    if not passages:
        st.info('No retrieved passages were saved.')
        return
    for p in passages:
        bits = [p['source']]
        if p['page'] is not None:
            bits.append(f"page {p['page']}")
        with st.expander(f"#{p['rank']} · " + ' · '.join(map(str, bits)), expanded=p['rank'] == 1):
            st.text(p['content'] or '(No text recorded)')


def question_detail(raw: dict, *, title: str | None = None) -> None:
    view = row_view(raw)
    if title:
        st.markdown(f'#### {title}')
    judge = raw.get('judge') or {}
    scorecard({**judge, **{k: raw.get(k) for k in RETRIEVAL_METRICS}},
              {**available([raw], RETRIEVAL_METRICS), **available([judge], ANSWER_METRICS)})
    missing = [k for k in raw.get('expected_keywords', []) if k.casefold() not in
               ' '.join(p['content'] for p in view['retrieved']).casefold()]
    if missing:
        st.warning('Expected keywords missing from the retrieved passages: ' + ', '.join(missing))
    left, right = st.columns(2)
    with left:
        st.markdown('**Generated answer**')
        st.write(view['answer'] or '(No answer recorded)')
        st.markdown('**Reference answer**')
        st.write(view['reference'] or '(No reference answer)')
        if view['feedback']:
            st.markdown('**Judge feedback**')
            st.write(view['feedback'])
    with right:
        st.markdown(f"**Passages the answer was based on ({len(view['retrieved'])})**")
        show_passages(view['retrieved'])


def per_question(experiment: dict, key: str) -> None:
    st.subheader('Question by question')
    st.caption('Find what goes wrong. **Diagnosis** says whether the search missed the right passages '
               '(fix retrieval: chunk size, top-K, query techniques) or the passages were found but the answer is '
               'still wrong (fix generation: answer model, prompt, re-ranking). Select a row to see the details.')
    rows = experiment.get('results', [])
    failures = {f['question']: f for f in experiment.get('failures', [])}
    if experiment.get('errors'):
        with st.expander(f"{len(experiment['errors'])} question(s) failed with an error", icon=':material/error:'):
            for err in experiment['errors']:
                st.markdown(f"**{err['question']}**")
                st.code(err['error'], language='text')
    if not rows:
        st.info('No saved question results.')
        return
    table = pd.DataFrame([{
        '#': i + 1,
        'Question': r.get('question', ''),
        'Category': r.get('category'),
        'Accuracy': (r.get('judge') or {}).get('accuracy'),
        'Keyword coverage': r.get('keyword_coverage'),
        'Diagnosis': ', '.join(FLAG_LABELS.get(f, f) for f in failures.get(r.get('question'), {}).get('flags', [])) or 'OK',
    } for i, r in enumerate(rows)]).dropna(axis=1, how='all')
    only_problems = st.toggle('Show only questions with a problem', key=f'problems_{key}')
    shown = table[table['Diagnosis'] != 'OK'] if only_problems else table
    picked = st.dataframe(shown, hide_index=True, on_select='rerun', selection_mode='single-row',
                          key=f'questions_{key}_{only_problems}', height=min(36 * (len(shown) + 1) + 2, 400),
                          column_config={'Keyword coverage': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent'),
                                         'Question': st.column_config.TextColumn(width='large')})
    if not picked.selection.rows:
        st.caption('Select a row to see the answer next to the passages it used.')
        return
    index = int(shown.iloc[picked.selection.rows[0]]['#']) - 1
    st.markdown(f"**Q{index + 1}. {rows[index].get('question', '')}**")
    question_detail(rows[index])


def compare_answers(report: dict, key: str) -> None:
    names = [e['name'] for e in report.get('experiments', [])]
    with st.expander('Compare two setups on the same question', icon=':material/compare_arrows:'):
        st.caption('See what a technique actually changed in the answer and the retrieved passages.')
        c1, c2 = st.columns(2)
        left = c1.selectbox('First setup', names, index=0, key=f'compare_left_{key}')
        right = c2.selectbox('Second setup', names, index=1, key=f'compare_right_{key}')
        pairs = pair_question_rows(report, left, right)
        if not pairs:
            st.info('No questions in common between these setups.')
            return
        i = st.selectbox('Question', range(len(pairs)), key=f'compare_question_{key}',
                         format_func=lambda n: f"{n + 1}. {pairs[n]['question'][:120]}")
        for column, name, raw in zip(st.columns(2), (left, right), (pairs[i]['left'], pairs[i]['right'])):
            with column:
                question_detail(raw, title=name)


def comparable_key(job: dict) -> tuple | None:
    """Runs with the same key used the same documents and questions, so they can be compared."""
    if job['kind'] == 'ask' and job.get('collection'):
        return job['collection'], 'asked'  # any questions asked in a workspace can be compared
    if job.get('collection') and job.get('test_set'):
        return job['collection'], job['test_set']
    if job['kind'] == 'financial':  # older FinanceBench runs, before they were linked to the collection
        from raglab_app import benchmarks
        collection = benchmarks.loaded('financebench')
        return (collection['slug'], 'financebench.jsonl') if collection else ('financebench',)
    return None


def run_entries(job: dict) -> list[dict]:
    """One entry per setup of a finished run, labelled by the run's name (plus the setup if it had several)."""
    path = Path(job['report'])
    report = load(job['id'], str(path), job['kind'], path.stat().st_mtime)
    experiments = report.get('experiments', [])
    name = (job.get('display_name') or 'Run').removeprefix('Ask: ')
    if job.get('number'):
        name = f"#{job['number']} · {name}"
    return [{'label': name if len(experiments) == 1 else f"{name} · {e['name']}", 'experiment': e, 'job': job}
            for e in experiments]


def short(text: str, length: int) -> str:
    text = ' '.join(str(text).split())
    return text if len(text) <= length else text[:length - 1].rstrip() + '\u2026'


def settings_of(entry: dict) -> dict:
    config, models = entry['experiment'].get('config', {}), entry['experiment'].get('models', {})
    return {'Chunk size': config.get('chunk_size', config.get('window_lines')),
            'Overlap': config.get('chunk_overlap', config.get('window_overlap')),
            'Top-K': config.get('top_k'), 'Embedding': models.get('embedding'),
            'Techniques': ('finance-tuned, ' if entry['job']['kind'] == 'financial' else '') + techniques(config),
            'Run on': entry['job']['created_at'][:16].replace('T', ' ')}


def rating_table(summaries: list[dict], metrics: dict):
    """Scores of each run, each cell shaded green / yellow / red by its Good / Fair / Poor rating."""
    labels = {key: label for key, (label, _) in metrics.items()}
    key_of = {label: key for key, label in labels.items()}
    data = pd.DataFrame([{labels[k]: s.get(k) for k in metrics} for s in summaries],
                        index=[s['name'] for s in summaries]).dropna(axis=1, how='all')
    shades = {'green': '#d4edda', 'orange': '#fff3cd', 'red': '#f8d7da'}

    def shade(column):
        key = key_of[column.name]
        return [f'background-color: {shades[rating(key, float(v))[1]]}; color: #1f1f1f' if pd.notna(v) else ''
                for v in column]
    return data.style.apply(shade).format(
        {c: (lambda v, k=key_of[c]: '' if pd.isna(v) else display(k, float(v))) for c in data.columns})


def compare_runs(entries: list[dict], key: str) -> None:
    """Side-by-side comparison of runs on the same test questions: settings, scores, plots, questions."""
    asked = list(dict.fromkeys(e['job'].get('question') for e in entries if e['job']['kind'] == 'ask'))
    question_ids = {q: f'Q{i}' for i, q in enumerate(asked, 1)}
    if len(asked) > 1:
        # Different questions: tag each run with its question's ID, or same-settings runs look identical.
        def tagged(e):
            number, _, rest = e['label'].partition(' · ')
            return f"{number} · {question_ids[e['job']['question']]} · {rest}"
        entries = [{**e, 'label': tagged(e)} if e['job']['kind'] == 'ask' and e['label'].startswith('#') else e
                   for e in entries]
    seen: dict[str, int] = {}
    unique = []
    for entry in entries:  # runs with identical settings get identical names: number them
        seen[entry['label']] = seen.get(entry['label'], 0) + 1
        unique.append({**entry, 'label': entry['label'] + (f" ({seen[entry['label']]})" if seen[entry['label']] > 1 else '')})
    entries = unique
    summaries = [{'name': e['label'], **e['experiment'].get('summary', {})} for e in entries]
    answer = available(summaries, ANSWER_METRICS)
    retrieval = retrieval_metrics_for(summaries)
    metrics = {**answer, **retrieval}

    if asked:
        st.markdown('**Questions**' if len(asked) > 1 else '**Question**')
        st.dataframe(pd.DataFrame([{'ID': question_ids[q], 'Question': q,
                                    'Runs': ', '.join(e['label'].split(' · ')[0] for e in entries
                                                      if e['job'].get('question') == q)} for q in asked]),
                     hide_index=True, column_config={'Question': st.column_config.TextColumn(width='large')})
    st.markdown('**Settings of each run**: what differs between them')
    st.dataframe(pd.DataFrame([{'Run': e['label'], **settings_of(e)} for e in entries]), hide_index=True)
    st.markdown('**Scores**: green good, yellow fair, red poor. Hover a column name for what it measures.')
    st.dataframe(rating_table(summaries, metrics),
                 column_config={label: st.column_config.Column(help=text) for label, text in metrics.values()})
    score_charts(summaries)

    rows = {}
    for entry in entries:
        for row in entry['experiment'].get('results', []):
            rows.setdefault(row.get('question', ''), {})[entry['label']] = row
    questions = list(rows)
    if not questions:
        return
    if len(questions) == 1:
        # One shared question: nothing to choose, show every run's answer straight away.
        st.markdown('**Answers side by side**')
        answers_side_by_side(entries, rows, questions[0], metrics)
        return

    st.markdown('**Question by question**: each question is a row, each run a column')
    if any(len(answered) < len(entries) for answered in rows.values()):
        st.caption('These runs answered different questions; an empty cell means that run did not ask that question.')
    options = {k: label for k, (label, _) in metrics.items()}
    if not options:
        return
    shown = st.selectbox('Score shown in the cells', list(options), format_func=options.get, key=f'qscore_{key}')

    def value(row):
        return (row.get('judge') or {}).get(shown) if shown in ANSWER_METRICS else row.get(shown)
    table = pd.DataFrame([{'Question': q[:120], **{e['label']: value(rows[q][e['label']]) if e['label'] in rows[q]
                                                  else None for e in entries}} for q in questions]).set_index('Question')
    shades = {'green': '#d4edda', 'orange': '#fff3cd', 'red': '#f8d7da'}
    styled = table.style.map(lambda v: f'background-color: {shades[rating(shown, float(v))[1]]}; color: #1f1f1f'
                             if pd.notna(v) else '').format(lambda v: '' if pd.isna(v) else display(shown, float(v)))
    picked = st.dataframe(styled, on_select='rerun', selection_mode='single-row', key=f'qtable_{key}')
    if not picked.selection.rows:
        st.caption('Select a question to see every run\'s answer and passages side by side.')
        return
    answers_side_by_side(entries, rows, questions[picked.selection.rows[0]], metrics)


def answers_side_by_side(entries: list[dict], rows: dict, question: str, metrics: dict) -> None:
    st.markdown(f'**{question}**')
    reference = next((r.get('reference_answer') for r in rows[question].values() if r.get('reference_answer')), '')
    if reference:
        st.caption(f'Correct answer: {reference}')
    answered = [e for e in entries if e['label'] in rows[question]]
    columns = st.columns(min(len(answered), 3))
    for i, entry in enumerate(answered):
        row = rows[question][entry['label']]
        with columns[i % len(columns)]:
            st.markdown(f"**{entry['label']}**")
            view = row_view(row)
            judge = row.get('judge') or {}
            st.markdown(score_badges({**judge, **row}, metrics))
            with st.container(border=True):
                st.write(view['answer'] or '(No answer recorded)')
            with st.expander(f"{len(view['retrieved'])} passages used"):
                for passage in view['retrieved']:
                    page = f" · page {passage['page']}" if passage['page'] is not None else ''
                    st.caption(f"#{passage['rank']} · {passage['source']}{page}")
                    st.text(passage['content'][:600])


def compare_evaluations(job: dict, others: list[dict]) -> None:
    key = comparable_key(job)
    candidates = [o for o in others if o['id'] != job['id'] and o.get('status') == 'completed'
                  and o.get('report') and Path(o['report']).is_file() and comparable_key(o) == key]
    st.subheader('Compare with other runs')
    st.caption('Runs on the same documents and test questions, side by side: before and after a change, or '
               'different settings and techniques.')
    if not key or not candidates:
        st.caption('No other finished run used the same documents and test questions yet.')
        return
    names = {o['id']: f"{o.get('display_name') or 'Run'} ({o['created_at'][:16].replace('T', ' ')})" for o in candidates}
    chosen = st.multiselect('Runs to add', list(names), format_func=names.get, key=f'compare_jobs_{job["id"]}')
    if chosen:
        entries = [e for item in [job] + [o for o in candidates if o['id'] in chosen] for e in run_entries(item)]
        compare_runs(entries, f'eval_{job["id"]}')


def render(job: dict, others: list[dict] | None = None, **options) -> None:
    """Load a job's report and show it. others: evaluations to offer for comparison (None hides it)."""
    report_path = Path(job['report'])
    report = load(job['id'], str(report_path), job['kind'], report_path.stat().st_mtime)
    render_report(report, job['id'], job=job, others=others, **options)


def render_report(report: dict, key: str, *, job: dict | None = None, others: list[dict] | None = None,
                  show_questions: bool = True, note_sample: bool = True) -> None:
    """The full evaluation view: comparison (several setups) or scorecards and charts (one setup),
    per-category breakdown, question by question and comparisons with other evaluations."""
    experiments = report.get('experiments', [])
    if not experiments:
        st.warning('The report contains no setups.')
        return
    collection = report.get('collection')
    if collection:
        asked = f"Question: **{collection['question']}**" if collection.get('question') else \
            f"Questions: **{collection.get('test_set')}**"
        st.caption(f"Documents: **{collection['name']}** ({len(collection['files'])} files) · {asked}")
    total = experiments[0].get('summary', {}).get('total') or len(experiments[0].get('results', []))
    if note_sample and total < 10:
        st.caption(f':material/info: Based on {total} test question(s). Averages over so few questions change a lot '
                   'from one question to the next; twenty or more give a fair picture.')
    if len(experiments) > 1:
        comparison(report)
        st.subheader('Details for one setup')
        name = st.selectbox('Setup', [e['name'] for e in experiments], key=f'setup_{key}')
        experiment = next(e for e in experiments if e['name'] == name)
    else:
        experiment = experiments[0]
        st.subheader('Scores')
    summary(experiment, report, charts=len(experiments) == 1)
    if len(experiments) == 1:
        settings_table(experiments)
    if show_questions:
        per_question(experiment, f"{key}_{experiment['name']}")
    if len(experiments) > 1:
        compare_answers(report, key)
    if job is not None and others is not None:
        compare_evaluations(job, others)


def single_answer_report(row: dict, *, config: dict, models: dict) -> dict:
    """Wrap one scored answer (raglab.evaluation.evaluate_result output) as a one-question report."""
    from raglab.failure import analyze_failure
    judge = row.get('judge') or {}
    summary = {key: row.get(key) for key in RETRIEVAL_METRICS}
    summary.update({key: judge.get(key) for key in ANSWER_METRICS})
    summary.update(total=1, successful=1, failed=0, latency_s=row.get('latency_s'))
    return {'experiments': [{'name': 'this answer', 'summary': summary, 'results': [row],
                             'failures': [analyze_failure(row)], 'errors': [], 'config': config, 'models': models}]}
