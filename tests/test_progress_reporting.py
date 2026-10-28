from raglab_app.jobs import progress_from_log
import json

def test_question_progress():
    event = dict(stage='evaluating', total_runs=2, completed_runs=1, current_run='hybrid', total_questions=10, question_done=4)
    p = progress_from_log('RAGLAB_PROGRESS '+json.dumps(event), 'financial', 'running')
    assert p['fraction'] == 0.7
    assert p['question_done'] == 4

def test_completed_and_bad_logs():
    assert progress_from_log('noise', 'financial', 'completed')['fraction'] == 1
    assert progress_from_log('RAGLAB_PROGRESS {bad json', 'financial', 'running')['fraction'] == 0
