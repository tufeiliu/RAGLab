import pytest

from raglab_app import settings

CHAT = {'mini': 'Mini', 'big': 'Big'}
EMBED = {'small': 'Small'}


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / 'settings.yaml'
    monkeypatch.setattr(settings, 'SETTINGS_FILE', path)
    return path


def test_values_from_the_file(settings_file):
    settings_file.write_text('defaults:\n  answer_model: big\n  chunk_size: 1200\n  chunk_overlap: 300\n'
                             '  top_k: 8\n  techniques: [rerank, opening]\n  judge: false\n'
                             'limits:\n  chunk_size: [100, 3000]\n')
    loaded = settings.load(CHAT, EMBED)
    assert (loaded.answer_model, loaded.embedding_model, loaded.chunk_size, loaded.chunk_overlap, loaded.top_k) == \
        ('big', 'small', 1200, 300, 8)
    assert loaded.techniques == ['rerank', 'opening'] and loaded.judge is False
    assert loaded.limits['chunk_size'] == (100, 3000) and not loaded.problems


def test_invalid_values_fall_back_with_a_message(settings_file):
    settings_file.write_text('defaults:\n  answer_model: missing\n  chunk_size: 900000\n  chunk_overlap: 5000\n'
                             '  techniques: [rerank, magic]\n  colour: blue\n')
    loaded = settings.load(CHAT, EMBED)
    assert loaded.answer_model == 'mini' and loaded.chunk_size == 800 and loaded.chunk_overlap < loaded.chunk_size
    assert loaded.techniques == ['rerank']
    text = ' '.join(loaded.problems)
    for fragment in ('"missing" is not in models.yaml', 'chunk_size must be', 'chunk_overlap must be',
                     'unknown techniques ignored: magic', 'unknown setting "colour"'):
        assert fragment in text


def test_missing_or_broken_file(settings_file):
    assert settings.load(CHAT, EMBED).chunk_size == 800  # no file: built-in defaults
    settings_file.write_text('defaults: [oops')
    loaded = settings.load(CHAT, EMBED)
    assert loaded.chunk_size == 800 and 'could not be read' in loaded.problems[0]


def test_shipped_settings_file_is_valid():
    from raglab_app.models import choices
    assert settings.load(choices('chat'), choices('embedding')).problems == []
