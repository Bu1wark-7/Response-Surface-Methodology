# -*- coding: utf-8 -*-
"""
整个流程只有 6 步，一眼能看完：
    1) 读取并清洗数据
    2) 按时间切分 训练 / 验证 / 测试
    3) 拟合 RSM 模型
    4) 预测并评价（含"补偿前 vs 补偿后"对比）
    5) 逐日误差（看跨天稳不稳）
    6) 导出结果
①	rsm_config.py	速读	只是个参数字典，认名字就行
②	rsm_main.py	必读	只看 6 步骨架，先别追细节
③	rsm_data.py	必读	比模型更重要——数据决定上限
④	rsm_model.py	★精读	全项目唯一真正的算法，只有 109 行
⑤	rsm_metrics.py	速读	理解"为什么不用 MSE"就够
⑥	rsm_export.py	速读	纯文件读写，没有算法
"""
import os

import numpy as np

from rsm_config import CFG
from rsm_data import load_data, split_by_time
from rsm_model import fit_model, predict, raw_formula
from rsm_metrics import evaluate, print_report, evaluate_by_day
from rsm_export import save_predictions, save_metrics, save_table, save_formula


def main():
    # 输出目录先建好
    os.makedirs(CFG["OUT_DIR"], exist_ok=True)
    print("=" * 60)
    print("RSM 响应面建模：出口光量子流量计补偿（真值 = 体积流量_首信）")
    print("=" * 60)

    # ---------------- 1. 读取并清洗 ----------------
    print("\n[1/6] 读取并清洗数据 ...")
    df = load_data(CFG)
    print(f"      有效样本 {len(df):,} 行")
    print(f"      时间范围 {df['time'].min()}  ~  {df['time'].max()}")

    # ---------------- 2. 按时间切分 ----------------
    print("\n[2/6] 按时间切分 训练 / 验证 / 测试 ...")
    train, valid, test = split_by_time(df, CFG)
    print(f"      训练 {len(train):,} 行（{train['time'].min()} ~ {train['time'].max()}）")
    print(f"      验证 {len(valid):,} 行")
    print(f"      测试 {len(test):,} 行")

    # ---------------- 3. 拟合 ----------------
    print("\n[3/6] 拟合 RSM 模型 ...")
    model = fit_model(train, CFG)
    print(f"      自变量 {model['features']}")

    # ---------------- 4. 预测并评价 ----------------
    print("\n[4/6] 预测并评价 ...")
    results, preds = {}, {}
    for name, part in [("训练集", train), ("验证集", valid), ("测试集", test)]:
        preds[name] = predict(model, part)
        m = evaluate(part["q_true"].values, preds[name])
        print_report(name, m)
        results[name] = m

    # ---- 补偿前 vs 补偿后（只在原表读数有效的样本上对比）----
    y_te = test["q_true"].values
    q_raw_te = test["q_raw"].values
    ok = ~np.isnan(q_raw_te)
    if ok.any():
        print(f"\n      —— 补偿效果对比（测试集，取两列都有效的 {ok.sum():,} 个样本）——")
        m_before = evaluate(y_te[ok], q_raw_te[ok])
        m_after = evaluate(y_te[ok], preds["测试集"][ok])
        print_report("补偿前（原表读数）", m_before)
        print_report("补偿后（RSM 预测）", m_after)

        gain_med = m_before["中位误差%"] - m_after["中位误差%"]
        gain_p90 = m_before["p90误差%"] - m_after["p90误差%"]
        ratio = m_before["中位误差%"] / m_after["中位误差%"] if m_after["中位误差%"] else float("inf")
        print(f"      → 中位误差改善 {gain_med:+.2f} 个百分点（{ratio:.1f} 倍）")
        print(f"      → p90  误差改善 {gain_p90:+.2f} 个百分点")

        results["补偿前_原表读数"] = m_before
        results["补偿后_RSM"] = m_after

    # ---------------- 5. 逐日误差 ----------------
    print("\n[5/6] 逐日误差（检查跨天稳定性）...")
    pred_test = preds["测试集"]
    daily = evaluate_by_day(test, pred_test, y_base=q_raw_te)
    print(daily.to_string(index=False))

    # ---------------- 6. 导出 ----------------
    print("\n[6/6] 导出结果 ...")
    save_predictions(os.path.join(CFG["OUT_DIR"], "预测结果_测试集.csv"), test, pred_test)
    save_metrics(os.path.join(CFG["OUT_DIR"], "指标汇总.csv"), results)
    save_table(os.path.join(CFG["OUT_DIR"], "逐日误差.csv"), daily)

    terms, intercept, centers = raw_formula(model)
    save_formula(os.path.join(CFG["OUT_DIR"], "RSM显式公式.txt"), terms, intercept, centers)

    # ---------------- 把公式也打印到屏幕上 ----------------
    print("\n" + "=" * 62)
    print("RSM 显式公式")
    print("=" * 62)
    print("第 1 步：变量中心化（减去训练集均值）")
    for label, mu in centers.items():
        print(f"        {label + '*':<8} = {label} - {mu:.6f}")
    print("\n第 2 步：代入公式")
    print(f"        Q̂_首信 = {intercept:.6f}")
    for name, coef in terms:
        sign = "+" if coef >= 0 else "-"
        print(f"                {sign} {abs(coef):.6f} * {name}")
    print("=" * 62)
    print(f"\n全部结果已保存到：{CFG['OUT_DIR']}")


if __name__ == "__main__":
    main()
