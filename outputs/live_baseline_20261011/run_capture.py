"""One authorized 24-request run; reuse the accepted collector and stop on failure."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('accepted_perf', ROOT / 'tests/run_performance_baseline.py')
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)
PRICES = Path(__file__).with_name('pricing.json')


def guarded_samples(base_url, selected, warmup_passes, formal_reps, requests_path, log=print):
    count = 0
    for phase, passes in (('warmup', warmup_passes), ('formal', formal_reps)):
        for rep in range(1, passes + 1):
            for scenario in selected:
                session_id = None
                for index, turn in enumerate(scenario['turns'], 1):
                    record = perf.run_one_request(base_url, session_id, turn['question'],
                                                  bool(turn.get('use_web_search')), turn['expected_path'])
                    record.update(request_id=f"{scenario['id']}-{phase}{rep}-t{index}",
                                  scenario_id=scenario['id'], scenario_name=scenario['name'],
                                  phase=phase, rep=rep, turn_index=index)
                    perf._append_jsonl(requests_path, record)
                    count += 1
                    log(f"{record['request_id']}: HTTP {record['http_status']}, {record['metrics']['status']}", flush=True)
                    session_id = record.get('server_session_id') or record.get('session_id')
                    # First warmup is also the smoke request; it is never repeated.
                    if record['http_status'] != 200 or record['metrics']['status'] != 'ok' or not session_id:
                        reason = {'request_id': record['request_id'], 'reason': 'request_failed_or_session_missing', 'sent': count}
                        requests_path.with_name('stop.json').write_text(json.dumps(reason, indent=2), encoding='utf-8')
                        return count
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    perf.PRICING = json.loads(PRICES.read_text(encoding='utf-8'))
    perf.run_samples = guarded_samples
    perf.APP_FILES_FOR_HASH += (str(Path(__file__).relative_to(ROOT)), str(PRICES.relative_to(ROOT)))
    result = perf.main(['collect', '--run-id', args.run_id, '--warmup', '1', '--reps', '3'])
    manifest_path = perf.PERF_ROOT / args.run_id / 'manifest.json'
    if not manifest_path.exists():
        return result or 1
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    requests = perf._read_jsonl(manifest_path.with_name('requests.jsonl'))
    # The accepted collector retains partial evidence; its exit code alone does not prove success.
    ok = (len(requests) == 24 and all(r['metrics']['status'] == 'ok' for r in requests)
          and manifest['isolation']['real_dirs_unchanged']
          and manifest['cleanup'].get('terminated') and manifest['cleanup'].get('port_released'))
    return 0 if result == 0 and ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
