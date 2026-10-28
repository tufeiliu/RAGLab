import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from financial_plan import load_financial_plan


def test_window_overrides_are_accepted(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("settings:\n  window_lines: 32\n  window_overlap: 10\nruns:\n  - name: test\n    settings:\n      window_lines: 48\n      window_overlap: 8\n")
    plan = load_financial_plan(path)
    assert plan["runs"][0]["settings"]["window_lines"] == 48


def test_bad_override_rejected(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("runs:\n  - name: test\n    settings:\n      mysterious_setting: 42\n")
    with pytest.raises(ValueError, match="mysterious_setting"):
        load_financial_plan(path)


def test_bad_overlap_rejected(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("runs:\n  - name: test\n    settings:\n      window_lines: 20\n      window_overlap: 20\n")
    with pytest.raises(ValueError, match="overlap"):
        load_financial_plan(path)
