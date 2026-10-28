from raglab_app import response_explorer as module


def test_passages_keep_order_metadata_and_optional_scores():
    passages = module.normalise_passages([
        {'content': 'one', 'metadata': {'source': '/path/A.txt', 'page': 7, 'chunk': 2}},
        {'content': 'two', 'metadata': {'source': '/path/B.txt'}, 'distance': 0.3},
    ])
    assert [x['rank'] for x in passages] == [1, 2]
    assert passages[0]['source'] == 'A.txt'
    assert passages[0]['page'] == 7
    assert passages[0]['score'] is None
    assert passages[1]['score_label'] == 'distance'


def test_pair_by_question_not_position():
    report = {'experiments': [
        {'name': 'baseline', 'results': [{'question': 'A', 'generated_answer': 'a'},
                                         {'question': 'B', 'generated_answer': 'b'}]},
        {'name': 'hybrid', 'results': [{'question': 'B', 'generated_answer': 'bb'},
                                       {'question': 'A', 'generated_answer': 'aa'}]},
    ]}
    pairs = module.pair_question_rows(report, 'baseline', 'hybrid')
    matched = {p['question']: p for p in pairs}
    assert matched['A']['right']['generated_answer'] == 'aa'
    assert matched['B']['left']['generated_answer'] == 'b'


def test_empty_or_missing_fields_safe():
    assert module.normalise_passages(None) == []
    view = module.row_view({'question': 'Why?', 'retrieved_documents': []})
    assert view['question'] == 'Why?'
    assert view['accuracy'] is None
    assert view['retrieved'] == []
