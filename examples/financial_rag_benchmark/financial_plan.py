"""FinanceBench-specific experiment plan validation.

The generic RAGLab loader only understands RAGConfig settings. Table-aware
window_lines/window_overlap belong to this *example*, not to core RAGLab.
"""
from pathlib import Path
import yaml

from raglab.experiments import SETTINGS, TECHNIQUES

FINANCIAL_SETTINGS = {"window_lines", "window_overlap", "judge_enabled"}
MODEL_ALIASES = {"chat", "embedding", "judge"}


def load_financial_plan(path):
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list) or not plan["runs"]:
        raise ValueError("Financial experiment YAML must contain a nonempty runs list")

    permitted = SETTINGS | MODEL_ALIASES | FINANCIAL_SETTINGS
    defaults = plan.get("settings", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError("settings must be a mapping")
    invalid = set(defaults) - permitted
    if invalid:
        raise ValueError(f"Unsupported default settings: {sorted(invalid)}")

    if "judge_enabled" in defaults and not isinstance(defaults["judge_enabled"], bool):
        raise ValueError("settings.judge_enabled must be true or false")

    for i, run in enumerate(plan["runs"]):
        if not isinstance(run, dict) or not isinstance(run.get("name"), str) or not run["name"]:
            raise ValueError(f"Run {i + 1} needs a nonempty name")
        settings = run.get("settings", {}) or {}
        techniques = run.get("techniques", {}) or {}
        if not isinstance(settings, dict) or not isinstance(techniques, dict):
            raise ValueError(f"Run {run['name']}: settings and techniques must be mappings")
        invalid = (set(settings) - permitted) | (set(techniques) - TECHNIQUES)
        if invalid:
            raise ValueError(f"Unsupported experiment overrides in {run['name']}: {sorted(invalid)}")
        if "judge_enabled" in settings and not isinstance(settings["judge_enabled"], bool):
            raise ValueError(f"Run {run['name']}: judge_enabled must be true or false")
        merged = {**defaults, **settings}
        width = int(merged.get("window_lines", 32))
        overlap = int(merged.get("window_overlap", 10))
        if not (8 <= width <= 300 and 0 <= overlap < width):
            raise ValueError(f"Run {run['name']}: window_lines must be 8–300 and overlap must be smaller")
    return plan
