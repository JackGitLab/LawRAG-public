import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.ask import ask, retrieve  # noqa: E402
from app.config import configure_utf8_stdio, log_current_exception  # noqa: E402


QUESTIONS_PATH = Path(__file__).with_name("regression_questions.json")


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)
    if not questions:
        raise ValueError("回归问题集不能为空")
    return questions


def node_label(item) -> str:
    metadata = item.node.metadata
    return f"{metadata['source']} {metadata['article']}"


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="运行 LawRAG 检索回归问题集")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="仅对空 expect 题调用 DeepSeek，额外验证是否拒答",
    )
    args = parser.parse_args()

    questions = load_questions()
    evaluated = 0
    hits = 0
    empty_count = 0
    refusal_hits = 0

    for case in questions:
        retrieved = retrieve(case["question"], top_k=3)
        labels = [node_label(item) for item in retrieved]
        expected = case["expect"]

        print(f"[{case['id']:02d}] 问题：{case['question']}")
        print("检索到的 top3 法条：")
        for rank, label in enumerate(labels, start=1):
            print(f"  {rank}. {label}")

        if expected:
            evaluated += 1
            matched = [label for label in expected if label in labels]
            if matched:
                hits += 1
                print(f"结果：命中（{', '.join(matched)}）")
            else:
                print(f"结果：未命中（期望：{', '.join(expected)}）")
        else:
            empty_count += 1
            print("结果：空 expect，仅统计，不判定拒答")
            if args.with_llm:
                answer, _ = ask(case["question"])
                refused = "现有资料无法回答" in answer
                refusal_hits += int(refused)
                print(f"DeepSeek 拒答：{'通过' if refused else '未通过'}")
        print()

    rate = hits / evaluated * 100 if evaluated else 0.0
    print("汇总：")
    print(f"题目总数：{len(questions)}")
    print(f"top3 命中率：{rate:.1f}%（{hits}/{evaluated}）")
    print(f"空 expect 题数量：{empty_count}")
    if args.with_llm:
        print(f"拒答通过率：{refusal_hits / empty_count * 100:.1f}%（{refusal_hits}/{empty_count}）")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        path = log_current_exception("regression", "回归测试失败")
        raise SystemExit(f"回归测试失败，完整日志已保存到：{path}") from None
