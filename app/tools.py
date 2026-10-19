"""本地确定性工具：法定赔偿/补偿计算器（不访问网络，不走 MCP）。"""

import math

from langchain_core.tools import tool

CAP_MULTIPLIER = 3
CAP_YEARS = 12


def fold_years(years: float) -> float:
    """按劳动合同法第四十七条折算年限：满 6 个月不满 1 年按 1 年，不满 6 个月按 0.5 年。

    折算的是“与月工资相乘的补偿月数”这一系数，而非实际工作年数：
    剩余部分大于 0 且不足半年记 0.5，达到半年但不足一年记 1。
    """
    full_years = math.floor(years)
    remainder = years - full_years
    if remainder == 0:
        return float(full_years)
    if remainder < 0.5:
        return full_years + 0.5
    return full_years + 1.0


@tool
def calculate_labor_compensation(
    monthly_wage: float,
    years: float,
    kind: str = "N",
    social_average_wage: float | None = None,
) -> str:
    """计算劳动合同解除的经济补偿金、代通知金或违法解除赔偿金。

    Args:
        monthly_wage: 解除劳动合同前十二个月的平均工资，单位元。
        years: 在本单位的工作年限，单位年，允许小数（3 年半填 3.5）。
        kind: N 为普通经济补偿；N+1 为经济补偿加一个月代通知金；2N 为违法解除赔偿金。
        social_average_wage: 用人单位所在地上年度职工月平均工资，单位元；不提供则不启用三倍封顶。
    """
    if monthly_wage <= 0:
        return "参数错误：缺少有效的月工资信息（必须大于 0）。请提供解除前十二个月的平均工资。"
    if years <= 0:
        return "参数错误：缺少有效的工作年限（必须大于 0）。"
    if kind not in {"N", "N+1", "2N"}:
        return "参数错误：kind 只能是 N、N+1 或 2N。"
    if social_average_wage is not None and social_average_wage < 0:
        return "参数错误：当地社平工资不能为负数。"

    folded_years = fold_years(years)
    wage = monthly_wage
    capped = social_average_wage is not None and monthly_wage > social_average_wage * CAP_MULTIPLIER
    if capped:
        wage = social_average_wage * CAP_MULTIPLIER
        folded_years = min(folded_years, CAP_YEARS)

    base = folded_years * wage
    if kind == "N+1":
        amount = base + wage
        formula = f"{folded_years:g} × {wage:.2f} + {wage:.2f}"
        basis = "第四十条、第四十六条、第四十七条；代通知金仅适用第四十条情形"
    elif kind == "2N":
        amount = 2 * base
        formula = f"2 × {folded_years:g} × {wage:.2f}"
        basis = "第四十七条、第八十七条；违法解除赔偿金按经济补偿标准的二倍计算"
    else:
        amount = base
        formula = f"{folded_years:g} × {wage:.2f}"
        basis = "第四十六条、第四十七条"

    cap_note = (
        "已按当地社平工资三倍封顶，且高工资情形计薪年限最高十二年。"
        if capped
        else "未启用三倍封顶，因未提供当地社平工资。"
        if social_average_wage is None
        else "月工资未超过当地社平工资三倍，未触发封顶。"
    )
    return (
        f"原始工作年限：{years:g} 年；折算后年限：{folded_years:g} 年；"
        f"计薪月工资：{wage:.2f} 元；计算公式：{formula}；"
        f"最终金额：{amount:.2f} 元。适用法条：{basis}。"
        f"前提说明：以输入信息属实且符合相应解除情形为前提；{cap_note}"
    )
