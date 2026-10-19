import argparse
import json
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.ask import ExternalServiceError, retrieve_with_strategy  # noqa: E402
from app.config import configure_utf8_stdio, log_current_exception  # noqa: E402


QUESTIONS_PATH = Path(__file__).with_name("regression_questions.json")
STRATEGIES = (
    "vector",
    "vector_rerank",
    "hyde",
    "multi_query",
    "bm25",
    "hybrid",
    "hybrid_rerank",
)
NOTES = {
    "vector": "纯向量 top3",
    "vector_rerank": "向量召回 10 条后重排",
    "hyde": "短假想条文向量召回后重排",
    "multi_query": "3 个短改写加原问题，RRF 合并后重排",
    "bm25": "jieba 分词后的纯 BM25 top3",
    "hybrid": "向量与 BM25 各召回 10 条，RRF 融合取 top3",
    "hybrid_rerank": "向量与 BM25 经 RRF 融合后重排取 top3",
}
GROUPS = (
    ("colloquial", "A 组（colloquial）"),
    ("terminology", "B 组（terminology）"),
)
RETRY_DELAY_SECONDS = 2


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def node_label(item) -> str:
    metadata = item.node.metadata
    return f"{metadata['source']} {metadata['article']}"


def retrieve_with_retry(question: str, strategy: str, top_k: int = 3):
    """长批次遇到临时外部服务错误时，仅重试当前题一次。"""
    try:
        return retrieve_with_strategy(question, strategy, top_k)
    except ExternalServiceError:
        time.sleep(RETRY_DELAY_SECONDS)
        return retrieve_with_strategy(question, strategy, top_k)


def group_result(cases, outcomes, strategy: str, question_type: str) -> dict:
    selected = [case for case in cases if case["type"] == question_type]
    hits = sum(outcomes[case["id"]][strategy] for case in selected)
    count = len(selected)
    return {
        "count": count,
        "hits": hits,
        "rate": hits / count * 100 if count else 0.0,
    }


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="运行 LawRAG 检索策略对比")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        help="只运行一个策略；不指定时运行全部七个策略",
    )
    args = parser.parse_args()
    strategies = (args.strategy,) if args.strategy else STRATEGIES
    questions = load_questions()
    inside_cases = [case for case in questions if case["expect"]]
    summaries = []
    outcomes = {case["id"]: {} for case in inside_cases}

    for strategy in strategies:
        hits = 0
        elapsed_ms = 0.0
        for case in questions:
            started = time.perf_counter()
            retrieved = retrieve_with_retry(case["question"], strategy, top_k=3)
            elapsed_ms += (time.perf_counter() - started) * 1000

            if not case["expect"]:
                continue
            labels = [node_label(item) for item in retrieved]
            hit = any(expected in labels for expected in case["expect"])
            outcomes[case["id"]][strategy] = hit
            if hit:
                hits += 1

        summaries.append(
            {
                "strategy": strategy,
                "inside_count": len(inside_cases),
                "hits": hits,
                "rate": hits / len(inside_cases) * 100,
                "average_ms": elapsed_ms / len(questions),
            }
        )
        print(
            f"完成 {strategy}：{hits}/{len(inside_cases)}",
            file=sys.stderr,
            flush=True,
        )

    print("## 检索策略对比")
    print()
    print("| 策略 | 库内题数 | top3 命中数 | top3 命中率 | 平均每题耗时(ms) | 备注 |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for item in summaries:
        print(
            f"| {item['strategy']} | {item['inside_count']} | {item['hits']} | "
            f"{item['rate']:.1f}% | {item['average_ms']:.1f} | "
            f"{NOTES[item['strategy']]} |"
        )

    print()
    print("## A/B 分组对比")
    print()
    group_headers = " | ".join(label for _, label in GROUPS)
    print(f"| 策略 | {group_headers} |")
    print(f"| --- | {' | '.join(['---:'] * len(GROUPS))} |")
    for strategy in strategies:
        cells = []
        for question_type, _ in GROUPS:
            result = group_result(inside_cases, outcomes, strategy, question_type)
            cells.append(
                f"{result['hits']}/{result['count']}（{result['rate']:.1f}%）"
            )
        print(f"| {strategy} | {' | '.join(cells)} |")

    print()
    print("## 逐题策略矩阵")
    print()
    strategy_headers = " | ".join(strategies)
    print(f"| ID | type | 问题 | 期望 | {strategy_headers} |")
    print(
        f"| ---: | --- | --- | --- | "
        f"{' | '.join(['---'] * len(strategies))} |"
    )
    for case in inside_cases:
        case_outcomes = outcomes[case["id"]]
        if all(case_outcomes.values()):
            continue
        cells = " | ".join(
            "命中" if case_outcomes[strategy] else "未命中"
            for strategy in strategies
        )
        print(
            f"| {case['id']} | {case['type']} | {case['question']} | "
            f"{', '.join(case['expect'])} | {cells} |"
        )

    all_failed = [
        case for case in inside_cases if not any(outcomes[case["id"]].values())
    ]
    print()
    print("## 所有策略均未命中")
    print()
    if not all_failed:
        print("无。")
    for case in all_failed:
        print(f"- [{case['id']}] {case['question']}（期望：{', '.join(case['expect'])}）")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        path = log_current_exception("strategy-comparison", "检索策略对比失败")
        raise SystemExit(f"检索策略对比失败，完整日志已保存到：{path}") from None
