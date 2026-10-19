import json
from pathlib import Path


QUESTIONS_PATH = Path(__file__).with_name("regression_questions.json")


def test_regression_question_set_shape() -> None:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    assert [case["id"] for case in questions] == list(range(1, len(questions) + 1))
    assert all(
        set(case) == {"id", "question", "expect", "type"} for case in questions
    )
    assert {case["type"] for case in questions} == {
        "colloquial",
        "terminology",
        "out_of_scope",
    }
    assert all(
        (case["type"] == "out_of_scope") == (not case["expect"])
        for case in questions
    )
    assert all(case["question"].strip() for case in questions)


def test_revised_multi_article_expectations() -> None:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = {case["id"]: case for case in json.load(file)}

    assert questions[31]["expect"] == [
        "中华人民共和国消费者权益保护法 第十八条",
        "中华人民共和国消费者权益保护法 第四十八条",
    ]
    assert questions[35]["expect"] == [
        "中华人民共和国消费者权益保护法 第八条",
        "中华人民共和国消费者权益保护法 第二十条",
    ]
