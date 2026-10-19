import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'outputs/live_baseline_20261011/run_capture.py'
spec = importlib.util.spec_from_file_location('live_capture', PATH)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def test_guarded_capture_preserves_24_slots_and_sessions(tmp_path, monkeypatch):
    calls = []
    def request(url, session, question, web, expected):
        calls.append((session, web))
        return {'http_status': 200, 'metrics': {'status': 'ok'},
                'server_session_id': session or f's{len(calls)}'}
    monkeypatch.setattr(live.perf, 'run_one_request', request)
    cases = live.perf.load_cases()['scenarios']
    output = tmp_path / 'requests.jsonl'
    assert live.guarded_samples('local', cases, 1, 3, output) == 24
    rows = live.perf._read_jsonl(output)
    assert len(rows) == 24
    assert sum(r['phase'] == 'warmup' for r in rows) == 6
    assert sum(session is None for session, _ in calls) == 16
    assert sum(web for _, web in calls) == 4
    assert len({r['request_id'] for r in rows}) == 24
    assert all(calls[i][0] is None for i, r in enumerate(rows) if r['turn_index'] == 1)
    assert all(calls[i][0] == rows[i-1]['server_session_id'] for i, r in enumerate(rows) if r['turn_index'] == 2)


def test_first_failure_is_saved_and_stops_without_retry(tmp_path, monkeypatch):
    calls = []
    def request(*args):
        calls.append(args)
        return {'http_status': 401, 'metrics': {'status': 'failed'}, 'server_session_id': None}
    monkeypatch.setattr(live.perf, 'run_one_request', request)
    output = tmp_path / 'requests.jsonl'
    assert live.guarded_samples('local', live.perf.load_cases()['scenarios'], 1, 3, output) == 1
    assert len(calls) == 1
    assert len(live.perf._read_jsonl(output)) == 1
    assert json.loads((tmp_path / 'stop.json').read_text())['sent'] == 1


def test_missing_session_stops_before_next_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(live.perf, 'run_one_request', lambda *a: {'http_status': 200, 'metrics': {'status': 'ok'}})
    assert live.guarded_samples('local', live.perf.load_cases()['scenarios'], 1, 3, tmp_path / 'requests.jsonl') == 1
