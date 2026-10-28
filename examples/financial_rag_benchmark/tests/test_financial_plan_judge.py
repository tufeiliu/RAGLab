import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from financial_plan import load_financial_plan


def test_judge_enabled_default_and_run_override(tmp_path):
    p = tmp_path / 'plan.yaml'
    p.write_text('settings:\n  judge_enabled: true\nruns:\n  - name: baseline\n    settings:\n      judge_enabled: false\n')
    plan = load_financial_plan(p)
    assert plan['settings']['judge_enabled'] is True
    assert plan['runs'][0]['settings']['judge_enabled'] is False


def test_judge_enabled_must_be_boolean(tmp_path):
    p = tmp_path / 'plan.yaml'
    p.write_text('settings:\n  judge_enabled: maybe\nruns:\n  - name: baseline\n')
    with pytest.raises(ValueError, match='judge_enabled'):
        load_financial_plan(p)
