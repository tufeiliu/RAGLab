from pathlib import Path
import pytest
from raglab.catalog import ModelCatalog

def test_missing_embedding_section(tmp_path):
    file = tmp_path / "models.yaml"
    file.write_text("models:\n  chat:\n    demo:\n      label: Demo\n      provider: openai\n      model: dummy\n      api_key_env: TEST_KEY\n  embedding:\n")
    assert ModelCatalog(file).get("chat", "demo", require_key=False).model == "dummy"
    assert ModelCatalog(file).choices("embedding", available_only=False) == {}

def test_bad_yaml_structure_gives_clear_error(tmp_path):
    file = tmp_path / "models.yaml"
    file.write_text("models:\n  chat: [bad]\n")
    with pytest.raises(ValueError, match="models.chat must be a mapping"):
        ModelCatalog(file)
