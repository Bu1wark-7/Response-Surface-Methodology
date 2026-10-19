# -*- coding: utf-8 -*-
"""
rsm_metrics.py —— 评价指标
==========================
为什么不用 MSE / 相关系数？
    因为真值里有少量极端值，MSE 会被它们主导；
    裸相关系数同样会被少数离群点毁掉。
    所以这里用"相对误差的中位数 / 平均 / p90"来评价，稳健得多。

对外函数：
    evaluate()          算一组指标，返回字典
    print_report()      把指标打印成一行
    evaluate_by_day()   按天分别算指标（看跨天稳不稳）
"""
import numpy as np
import pandas as pd


def evaluate(y_true, y_pred):
    """计算 R²、中位误差、平均误差、p90 误差。返回字典。

    相对误差 = |预测 - 真值| / 真值 * 100%
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rel = np.abs(y_pred - y_true) / y_true #相对差

    # R² = 1 - 残差平方和 / 总平方和
    # 若真值完全没有波动（总平方和 = 0），R² 没有意义，返回 nan
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) #偏移量的平方和
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    return {
        "R2": round(r2, 4),#保留两位小数
        "中位误差%": round(float(np.median(rel) * 100), 2), #np.median 中位数
        "平均误差%": round(float(np.mean(rel) * 100), 2),
        "p90误差%": round(float(np.percentile(rel, 90) * 100), 2),#九成样本在np.percentile误差内
    }


def print_report(name, m):
    """把指标打印成一行，方便逐个数据集对照。"""
    print(f"   [{name}] R²={m['R2']:.4f}   "
          f"中位误差={m['中位误差%']:.2f}%   "
          f"平均误差={m['平均误差%']:.2f}%   "
          f"p90={m['p90误差%']:.2f}%")


def evaluate_by_day(df, y_pred, y_base=None, min_samples=30):
    """按"天"分别算误差，返回一张 DataFrame。

    作用：检查模型跨天稳不稳。
    如果某几天误差特别大，就说明存在"日间漂移"问题，需要留意。

    y_base：可选的"补偿前基准"（原表读数）。
            传了它就多一列 `原表中位误差%`，方便逐天看补偿提升了多少。
    min_samples：当天样本少于这个数就跳过（样本太少的日统计量没意义）。
    注意 df 里必须有 time 和 q_true 两列。
    """
    tmp = pd.DataFrame({
        "day":  df["time"].dt.floor("D").values,
        "y":    df["q_true"].values,
        "pred": np.asarray(y_pred, dtype=float),
    })
    if y_base is not None:
        tmp["base"] = np.asarray(y_base, dtype=float)

    rows = []
    for day, g in tmp.groupby("day"):
        if len(g) < min_samples:
            continue
        m = evaluate(g["y"].values, g["pred"].values)
        m["日期"] = str(day)[:10]
        m["样本数"] = len(g)
        if y_base is not None:
            m["原表中位误差%"] = round(float(np.nanmedian(
                np.abs(g["base"].values - g["y"].values) / g["y"].values) * 100), 2)
        rows.append(m)

    cols = ["日期", "样本数", "R2", "中位误差%", "平均误差%", "p90误差%"]
    if y_base is not None:
        cols.append("原表中位误差%")
    return pd.DataFrame(rows)[cols]
