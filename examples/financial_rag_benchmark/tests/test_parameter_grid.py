import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ui_backend as ui

def test_parse_values():
    assert ui.parse_integer_values('300, 500, 300', label='lines', minimum=8, maximum=1000)==[300,500]
    with pytest.raises(ValueError): ui.parse_integer_values('5, abc', label='k', minimum=1, maximum=50)
    with pytest.raises(ValueError): ui.parse_integer_values('0', label='k', minimum=1, maximum=50)

def test_grid_individual_settings(monkeypatch):
    monkeypatch.setattr(ui, 'model_choices', lambda k:[('Mini','chat') ] if k=='chat' else [('Small','small'),('Large','large')])
    plan=ui.make_grid_plan(chat='chat',judge='chat',judge_enabled=False,
        embeddings=['small','large'],top_ks='5',window_lines='24, 32',window_overlaps='4',
        retrievals=['vector'],techniques=[],compare_baseline=False,mode='grid')
    assert len(plan['runs'])==4
    assert {r['settings']['window_lines'] for r in plan['runs']}=={24,32}
    assert {r['settings']['embedding'] for r in plan['runs']}=={'small','large'}

def test_single_rejects_multivalues(monkeypatch):
    monkeypatch.setattr(ui,'model_choices',lambda k:[('One','chat')] if k=='chat' else [('Small','small')])
    with pytest.raises(ValueError,match='Single mode'):
        ui.make_grid_plan(chat='chat',judge='chat',judge_enabled=True,embeddings=['small'],
          top_ks='5, 10',window_lines='32',window_overlaps='4',retrievals=['vector'],
          techniques=[],compare_baseline=False,mode='single')
