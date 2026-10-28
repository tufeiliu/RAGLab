"""Regression: experiment summaries, not stale leaderboard copies, are authoritative."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('financial_ui_backend_test', ROOT / 'ui_backend.py')
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


def test_leaderboard_refreshes_from_summary(tmp_path, monkeypatch):
    result_dir = tmp_path / 'results'
    result_dir.mkdir()
    monkeypatch.setattr(ui, 'RESULTS', result_dir)
    report = {
        'experiments': [{'name': 'baseline', 'models': {}, 'summary': {'mrr': 0.5, 'accuracy': 3.0},
                         'results': [{'question': 'Q', 'mrr': 0.5}]}],
        'leaderboard': [{'name': 'baseline', 'mrr': None, 'accuracy': 3.0}],
    }
    report_path = result_dir / 'report.json'
    report_path.write_text(json.dumps(report))
    loaded = ui.load_report(report_path)
    assert loaded['leaderboard'][0]['mrr'] == 0.5
