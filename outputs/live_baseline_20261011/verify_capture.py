"""Execution-side checks of saved live evidence; this is NOT independent acceptance."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'logs/performance/perf-20261011-live1'
OUT = Path(__file__).parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    rows = read_rows(RUN / 'requests.jsonl')
    observations = read_rows(RUN / 'observations.jsonl')
    checks = {}
    checks['request_count_24'] = len(rows) == 24
    checks['phases_6_18'] = Counter(r['phase'] for r in rows) == {'warmup': 6, 'formal': 18}
    checks['all_http_200'] = all(r['http_status'] == 200 for r in rows)
    checks['all_path_match'] = all(r.get('path_match') for r in rows)
    checks['no_unassigned_observations'] = manifest['observation_unassigned'] == 0
    checks['real_data_unchanged'] = manifest['isolation']['snapshot_before'] == manifest['isolation']['snapshot_after']
    checks['own_process_cleaned'] = bool(manifest['cleanup'].get('terminated') and manifest['cleanup'].get('port_released'))
    checks['price_snapshot_matches'] = manifest['pricing'] == json.loads((OUT / 'pricing.json').read_text(encoding='utf-8'))
    timing_ok = []
    tasks = defaultdict(list)
    samples = defaultdict(list)
    for r in rows:
        events = r['events']
        tokens = [e for e in events if e['type'] == 'token' and e['text_nonempty']]
        dones = [e for e in events if e['type'] == 'done']
        valid = bool(tokens and len(dones) == 1 and not r['failure']
                     and not any(e['type'] == 'error' for e in events)
                     and all(e['order'] < dones[0]['order'] for e in tokens))
        valid = valid and abs(tokens[0]['t'] * 1000 - r['metrics']['first_text_ms']) <= 0.002
        valid = valid and abs(dones[0]['t'] * 1000 - r['metrics']['done_ms']) <= 0.002
        timing_ok.append(valid)
        tasks[(r['scenario_id'], r['phase'], r['rep'])].append(r)
        if r['phase'] == 'formal':
            samples[f"{r['scenario_id']}/T{r['turn_index']}/first_ms"].append(tokens[0]['t'] * 1000)
            samples[f"{r['scenario_id']}/T{r['turn_index']}/done_ms"].append(dones[0]['t'] * 1000)
    checks['raw_timing_recomputed'] = len(timing_ok) == 24 and all(timing_ok)
    checks['session_isolation_16'] = len(tasks) == 16 and len({rs[0]['session_id'] for rs in tasks.values()}) == 16
    checks['within_task_same_session'] = all(len({r['session_id'] for r in rs}) == 1 for rs in tasks.values())
    for (sid, phase, rep), rs in tasks.items():
        if phase == 'formal':
            samples[f'{sid}/task_done_sum_ms'].append(sum(next(e['t'] for e in r['events'] if e['type'] == 'done') * 1000 for r in rs))
    reranks = [r for r in observations if r['kind'] == 'retrieval' and r['name'] == 'rerank']
    llms = [r for r in observations if r['kind'] == 'llm']
    checks['rerank_attempts_recorded'] = bool(reranks) and all(isinstance(r.get('attempts'), int) and r['attempts'] >= 1 for r in reranks)
    old = json.loads((OUT / 'protected_before.json').read_text(encoding='utf-8'))
    checks['protected_unchanged'] = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in old.items())
    summary = {'checks': checks, 'all_checks_pass': all(checks.values()), 'request_count': len(rows),
               'observations': len(observations), 'llm_calls': len(llms),
               'models': dict(Counter(r.get('model') for r in llms)),
               'rerank_calls': len(reranks), 'rerank_attempts': [r['attempts'] for r in reranks],
               'formal_stats': {k: {'n':len(v), 'min':min(v), 'median':median(v), 'max':max(v)} for k,v in samples.items()}}
    (OUT / 'execution_checks.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
