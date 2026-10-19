"""OPT1 优化后真实对照复测的专用采集入口（薄包装器，真实执行需另行授权）。

复用：
- 已验收采集器 tests/run_performance_baseline.py（经 outputs/live_baseline_20261011/run_capture.py
  的同一模块实例导入，采集、隔离、清理、报告与统计公式零改动）；
- 已验收首错停止逻辑 guarded_samples（直接复用该函数：首个请求失败或缺 session 即落盘停止，
  不重试整轮；首条预热同时作为冒烟，不额外增加请求）。

显式选用优化后对照场景 tests/performance_cases_opt1.json：本包装器把采集器的 CASES_PATH
指向该文件，**实际 collect 读取的就是它**（manifest 的 cases_sha256/cases_file_sha256 与
场景身份均来自该文件；不是只在文档或 validate 阶段指定）。

不修改 OPT1 图实现、计算器、模型、提示词、检索策略或测量统计公式；
不覆盖旧包装器，不扩建通用采集框架；E-102 另案保留。
CLI 退出码不等于 24 条请求全部成功：结果以 requests.jsonl 与 manifest 证据为准。
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE1_WRAPPER = ROOT / 'outputs' / 'live_baseline_20261011' / 'run_capture.py'
CASES = ROOT / 'tests' / 'performance_cases_opt1.json'
PRICES = Path(__file__).with_name('pricing.json')

# 复用旧包装器模块：live1.perf 即已验收采集器的同一模块实例；
# live1.guarded_samples 即已验收首错停止逻辑（不复制、不重写）。
_spec = importlib.util.spec_from_file_location('live1_capture', LIVE1_WRAPPER)
live1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live1)
perf = live1.perf
guarded_samples = live1.guarded_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    # PRE-01：实际 collect 读取优化后对照场景（含场景身份与冻结计划同旧版一致）
    perf.CASES_PATH = CASES
    # 本次价格依据（含未核实项）：随 manifest 冻结，不使用代码内默认价格
    perf.PRICING = json.loads(PRICES.read_text(encoding='utf-8'))
    # 首错落盘即停止；首条预热即冒烟，成功路径不重复冒烟
    perf.run_samples = guarded_samples
    # PRE-02：manifest 记录实际场景文件、本包装器、价格文件与复用来源（旧包装器）的身份哈希
    perf.APP_FILES_FOR_HASH += (
        CASES.relative_to(ROOT).as_posix(),
        Path(__file__).relative_to(ROOT).as_posix(),
        PRICES.relative_to(ROOT).as_posix(),
        LIVE1_WRAPPER.relative_to(ROOT).as_posix(),
    )
    result = perf.main(['collect', '--run-id', args.run_id, '--warmup', '1', '--reps', '3'])
    manifest_path = perf.PERF_ROOT / args.run_id / 'manifest.json'
    if not manifest_path.exists():
        return result or 1
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    requests = perf._read_jsonl(manifest_path.with_name('requests.jsonl'))
    # 采集器保留部分证据；退出码本身不能证明 24 条全部成功，逐条核对落盘证据。
    # 清理成功以 cleanup_complete（E-102 R1：父进程＋后代＋端口全部确认）为准，
    # 父进程 terminated 单独为真不等于整个采集进程树清理完成。
    ok = (len(requests) == 24 and all(r['metrics']['status'] == 'ok' for r in requests)
          and manifest['isolation']['real_dirs_unchanged']
          and manifest['cleanup'].get('cleanup_complete') is True)
    return 0 if result == 0 and ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
