import math

import pytest

from app.tools import calculate_labor_compensation, fold_years


@pytest.mark.parametrize(
    ("years", "expected"),
    [(5 / 12, 0.5), (6 / 12, 1.0), (3.5, 4.0), (3 + 5 / 12, 3.5)],
)
def test_fold_years_boundaries(years, expected) -> None:
    assert fold_years(years) == expected


@pytest.mark.parametrize(
    ("years", "expected"),
    [
        (0.04, 0.5),
        (0.49, 0.5),
        (0.5, 1.0),
        (0.99, 1.0),
        (1.0, 1.0),
        (1.01, 1.5),
        (1.49, 1.5),
        (1.5, 2.0),
        (3.5, 4.0),
    ],
)
def test_fold_years_conversion(years, expected) -> None:
    assert fold_years(years) == expected


@pytest.mark.parametrize(
    ("years", "expected"),
    [
        # 半年边界：略小于半年不提前进位，恰好半年计 1，略大于半年计 1
        (math.nextafter(0.5, 0.0), 0.5),
        (0.5, 1.0),
        (math.nextafter(0.5, 1.0), 1.0),
        # 整年边界：精确整年不额外 +0.5，刚超过整年保留剩余 0.5
        (math.nextafter(1.0, 0.0), 1.0),
        (1.0, 1.0),
        (math.nextafter(1.0, 2.0), 1.5),
        # 1.5 半年边界前后
        (math.nextafter(1.5, 0.0), 1.5),
        (1.5, 2.0),
        (math.nextafter(1.5, 2.0), 2.0),
        # 2.0 整年边界前后
        (math.nextafter(2.0, 0.0), 2.0),
        (2.0, 2.0),
        (math.nextafter(2.0, 3.0), 2.5),
    ],
)
def test_fold_years_float_boundaries(years, expected) -> None:
    assert fold_years(years) == expected


def test_fold_years_conversion_is_non_decreasing() -> None:
    # 用工龄单调递增的固定样本，折算系数不得下降
    samples = [0.01, 0.04, 0.49, 0.5, 0.99, 1.0, 1.01, 2.5, 3.5, 5.0, 9.5, 12.0, 15.0]
    results = [fold_years(y) for y in samples]
    for lower, upper in zip(results, results[1:]):
        assert upper >= lower


@pytest.mark.parametrize(
    ("years", "folded_text", "amount"),
    [
        (0.04, "0.5", "4000.00"),
        (0.49, "0.5", "4000.00"),
        (0.5, "1", "8000.00"),
        (1.0, "1", "8000.00"),
        (1.01, "1.5", "12000.00"),
        (1.5, "2", "16000.00"),
    ],
)
def test_calculate_labor_compensation_fold_text_consistency(
    years, folded_text, amount
) -> None:
    result = calculate_labor_compensation.invoke(
        {"monthly_wage": 8000, "years": years, "kind": "N"}
    )
    assert f"折算后年限：{folded_text} 年" in result
    assert f"计算公式：{folded_text} × 8000.00；" in result
    assert f"最终金额：{amount} 元" in result


@pytest.mark.parametrize(
    ("years", "kind", "amount"),
    [(3.5, "N", "32000.00"), (3.5, "N+1", "40000.00"), (3.5, "2N", "64000.00"),
     (5 / 12, "N", "4000.00"), (6 / 12, "N", "8000.00")],
)
def test_calculate_labor_compensation_amounts(years, kind, amount) -> None:
    result = calculate_labor_compensation.invoke(
        {"monthly_wage": 8000, "years": years, "kind": kind}
    )
    assert f"最终金额：{amount} 元" in result


@pytest.mark.parametrize(
    ("years", "amount"),
    [(10, "60000.00"), (15, "72000.00")],
)
def test_calculate_labor_compensation_caps_high_wage(years, amount) -> None:
    result = calculate_labor_compensation.invoke(
        {
            "monthly_wage": 8000,
            "years": years,
            "kind": "N",
            "social_average_wage": 2000,
        }
    )
    assert f"最终金额：{amount} 元" in result


@pytest.mark.parametrize(
    "arguments",
    [
        {"monthly_wage": -1, "years": 1},
        {"monthly_wage": 8000, "years": -1},
        {"monthly_wage": 8000, "years": 1, "kind": "wrong"},
    ],
)
def test_calculate_labor_compensation_invalid_parameters_return_text(arguments) -> None:
    result = calculate_labor_compensation.invoke(arguments)
    assert result.startswith("参数错误：")


@pytest.mark.parametrize(
    "arguments",
    [
        {"monthly_wage": 0, "years": 1},
        {"monthly_wage": 8000, "years": 0},
        {"monthly_wage": -1, "years": 1},
        {"monthly_wage": 8000, "years": -1},
    ],
)
def test_non_positive_calculation_parameters_never_return_fake_amount(arguments) -> None:
    result = calculate_labor_compensation.invoke(arguments)
    assert result.startswith("参数错误：")
    assert "0.00 元" not in result
    assert "最终金额" not in result
