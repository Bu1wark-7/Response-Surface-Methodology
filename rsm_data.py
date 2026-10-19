# -*- coding: utf-8 -*-
"""
rsm_data.py —— 读取数据 + 清洗 + 切分
=====================================
只做三件事：
    load_data()     读 CSV，清洗，返回一张      干净的表
    split_by_time() 按时间先后切成 训练 / 验证 / 测试
"""
import numpy as np
import pandas as pd


def load_data(cfg):
    """读 CSV → 清洗 → 返回干净的 DataFrame。

    清洗步骤（顺序不能乱）：
        1) 把无效码（-1、-1111、-599900 …）置为 NaN
        2) 过滤掉停泵、停排量、压差为 0 的无效段
        3) ★ 卡掉真值的极端尖峰（0 < 真值 < 60）
        4) 算出派生变量 √Δp

    注意：q_raw（原来的光量子读数）只做无效码清洗，**不参与筛选**，
          它仅用于在结果里和补偿后的预测值并排对比。
    """
    # --- 1. 只读取需要的列，能快很多 ---
    #     注意：pandas 的 usecols 会按"文件里的顺序"返回列，
    #     不一定等于我们写的顺序，所以必须"按名字"改名，不能"按位置"改名。
    rename_map = {cn: en for en, cn in cfg["COLS"].items()}   # 中文名 -> 英文短名
    en_names = list(cfg["COLS"].keys())

    df = pd.read_csv(cfg["CSV"], usecols=list(rename_map.keys()),  #把 170 万行 CSV 变成表格对象
                     nrows=cfg["MAX_ROWS"], encoding="utf-8-sig")
    df = df.rename(columns=rename_map)[en_names]                   # 改完名再统一排好顺序
    df["time"] = pd.to_datetime(df["time"])

    # --- 2. 无效码 → NaN ---
    #     这些列的正常值都 >= 0，所以凡 <= SENTINEL 的都是无效码
    value_cols = ["dp", "p", "temp", "q_in", "q_true", "q_raw",
                  "pump1", "pump2", "pump3"]
    for c in value_cols:
        df.loc[df[c] <= cfg["SENTINEL"], c] = np.nan

    # --- 3. 泵冲合计（要 > 0 才算在正常钻进）---
    df["pump"] = df[["pump1", "pump2", "pump3"]].sum(axis=1, min_count=1)

    # --- 4. 过滤出一条一条都合法的记录 ---
    ok = (
        df["dp"].notna() & (df["dp"] > 0)                      # 压差有效
        & df["p"].notna() & (df["p"] > 0)                      # 压力有效
        & df["temp"].notna()                                   # 温度有效
        & df["q_in"].notna() & (df["q_in"] > cfg["Q_IN_MIN"])  # 入口流量有效
        & df["q_true"].notna()
        & (df["q_true"] > cfg["Q_TRUE_MIN"])
        & (df["q_true"] < cfg["Q_TRUE_MAX"])                   # ★ 卡尖峰
        & df["pump"].notna() & (df["pump"] > 0)                # 泵在转
    )
    df = df.loc[ok].sort_values("time").reset_index(drop=True)

    # --- 5. 派生变量：√Δp（差压式流量计的物理内核）---
    df["sq_dp"] = np.sqrt(df["dp"])  #根号进入变成线性

    return df


def split_by_time(df, cfg):
    """按时间先后切分成 (训练集, 验证集, 测试集)。

    注意：一定按时间顺序切，不能随机打乱！
    因为流量是连续变化的，相邻点高度相似；
    如果随机打乱，训练集里会有测试集"隔壁"的点，造成数据泄漏、
    指标虚高，上线后完全失效。
    """
    n = len(df)
    i1 = int(n * cfg["TRAIN_RATIO"])
    i2 = int(n * (cfg["TRAIN_RATIO"] + cfg["VALID_RATIO"]))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()
