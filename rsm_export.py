# -*- coding: utf-8 -*-
"""
rsm_export.py —— 导出结果
=========================
只负责把结果写到 outputs 文件夹里。
对外 4 个函数：
    save_predictions()  逐点预测结果.csv
    save_metrics()      指标汇总.csv
    save_table()        任意表格.csv
    save_formula()      RSM显式公式.txt
"""
import os
import numpy as np
import pandas as pd


def save_predictions(file_path, df, y_pred):
    """保存逐点预测结果，并把「原来的流量计读数」并排放进来。

    列：
        时间 | 真值_首信 | 光量子读数 | 预测_RSM |
        原表绝对误差 | 原表相对误差% | 补偿后绝对误差 | 补偿后相对误差%

    这样一眼就能看出补偿到底有没有用：
        原表相对误差%  = 不补偿直接读数的误差
        补偿后相对误差% = 用 RSM 补偿后的误差
    """
    y = df["q_true"].values
    # 原表读数可能缺失（该行没读到），缺失时整列留空
    if "q_raw" in df.columns:
        q_raw = df["q_raw"].values
    else:
        q_raw = np.full(len(df), np.nan)

    out = pd.DataFrame({
        "时间":            df["time"].values,
        "真值_首信":        y,
        "光量子读数":       q_raw,
        "预测_RSM":        y_pred,
        "原表绝对误差":     q_raw - y,
        "原表相对误差%":    (q_raw - y) / y * 100,
        "补偿后绝对误差":   y_pred - y,
        "补偿后相对误差%":  (y_pred - y) / y * 100,
    })
    out.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"   已保存 {os.path.basename(file_path)}  （{len(out):,} 行）")


def save_metrics(file_path, results):
    """保存指标汇总。

    results 形如 {"训练集": {...}, "验证集": {...}, "测试集": {...}}
    转成 DataFrame 后，每一行是一个数据集。
    """
    df = pd.DataFrame(results).T
    df.index.name = "数据集"
    df.to_csv(file_path, encoding="utf-8-sig")
    print(f"   已保存 {os.path.basename(file_path)}")


def save_table(file_path, df):
    """保存任意 DataFrame。"""
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"   已保存 {os.path.basename(file_path)}")


def save_formula(file_path, terms, intercept, centers):
    """把显式公式写成 txt，可以直接抄进论文。

    形式：
        先列出中心化要减掉的均值，
        再写 Q̂ = b0 + a1*√Δp* + a2*P* + ...
    """
    lines = [
        "RSM 显式公式（系数已在原始变量上，可直接使用）",
        "=" * 62,
        "",
        "第 1 步：先把变量中心化（减去训练集均值）",
    ]
    for label, mu in centers.items():
        lines.append(f"        {label + '*':<8} = {label} - {mu:.6f}")

    lines += [
        "",
        "第 2 步：代入下面的公式",
        f"        Q̂_首信 = {intercept:.6f}",
    ]
    for name, coef in terms:
        sign = "+" if coef >= 0 else "-"
        lines.append(f"                {sign} {abs(coef):.6f} * {name}")

    lines += [
        "",
        "-" * 62,
        "变量说明：",
        "  √Δp   = sqrt(压差_出口光量子)    单位 [kPa^0.5]",
        "  P     = 压力_出口光量子          单位 [MPa]",
        "  T     = 温度_出口光量子          单位 [°C]",
        "  Q_in  = 入口流量                 单位 [L/s]",
        "  带 * 的变量表示已经减去了上面的均值",
        "",
        "  Q̂_首信 的单位为 [L/s]",
        "",
        "说明：此处的常数项 = 所有自变量取训练集均值时的预测值。",
    ]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   已保存 {os.path.basename(file_path)}")
