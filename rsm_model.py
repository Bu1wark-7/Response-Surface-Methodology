# -*- coding: utf-8 -*-
"""
rsm_model.py —— RSM 响应面模型
==============================
响应面法（Response Surface Methodology）本质就是：
    把几个自变量做 2 次多项式展开，再用线性回归拟合。

本例的自变量只有 4 个：
    √Δp、P、T、Q_in
二次展开后得到 14 个基函数（不含常数项）：
    4 个一次项 + 4 个平方项 + 6 个交叉项
最后 y = 常数 + 这 14 项的加权和。

对外只暴露 3 个函数：
    fit_model()    拟合
    predict()      预测
    raw_formula()  把模型还原成"原始变量上的显式公式"
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

# 参与回归的 4 个自变量（对应 df 里的列名）
FEATURES = ["sq_dp", "p", "temp", "q_in"]

# 打印公式时用的好看名字（和 FEATURES 一一对应）
FEATURE_LABELS = ["√Δp", "P", "T", "Q_in"]


def fit_model(train, cfg):
    """在训练集上拟合 RSM，返回一个字典（装着拟合好的模型）。

    整个过程分 4 步：
        1) 中心化：每个自变量减去它在训练集上的均值
        2) 二次多项式展开：(n, 4) → (n, 14)
        3) 标准化：每个基函数变成均值 0、标准差 1
        4) 岭回归求解系数
    """
    X = train[FEATURES].values          # 自变量矩阵 (n, 4)
    y = train["q_true"].values          # 目标真值    (n,)

    # --- 1. 中心化 ---
    center = X.mean(axis=0)     #.mean(0)-按列求平均值，也就是期望 E[.]
    Xc = X - center

    # --- 2. 二次多项式展开 ---
    poly = PolynomialFeatures(degree=cfg["POLY_DEGREE"], include_bias=False)
    Z = poly.fit_transform(Xc)          #标准响应面二阶模型(不包含系数)
    #poly.fit_transform([x1,x2,x3,x4]) = [ x1, x2, x3, x4, 
    #  14个基函数                           x1², x2², x3², x4², 
    #                                      x1x2, x1x3, x1x4, x2x3, x2x4, x3x4 ]

    # --- 3. 标准化 ---
    # 目的：让岭回归的正则惩罚对所有基函数一视同仁
    scaler = StandardScaler() #创建标准化转换器
    Zs = scaler.fit_transform(Z)
    # 计算:E[.]  标准差_s = (.mean(离均差^2))^1/2  标准化数据表元素 Zs_ij = (Z_ij-E_j) /s_j  
    # 相关系数矩阵R = 1/n * Zs.T * Zs

    # --- 4. 岭回归求解系数 ---
    ridge = Ridge(alpha=cfg["RIDGE_ALPHA"])
    # LossFunction = 残差平方和 + 加权(alpha)系数和 → 平方范数
    # 然后对L求系数的梯度并令其为0求解
    ridge.fit(Zs, y) 
    # return include ridge.intercept_(RealNumber)  ridge.coef_(shape(14,1))_betaVector

    # return dict
    # poly:二次展开器 power表
    # scaler：mean_ scaler_
    # ridge:标准化系数beta intercept_
    
    return {"poly": poly, "scaler": scaler, "ridge": ridge,"features": FEATURES, "center": center}


def predict(model, df):
    """
    用拟合好的模型对 df 做预测，返回预测值（numpy 数组）。

    变换顺序必须和 fit_model 里完全一致：
        中心化 → 多项式展开 → 标准化 → 岭回归预测
    """
    X = df[model["features"]].values - model["center"]      # 中心化
    Z = model["poly"].transform(X)                          # 多项式展开
    Zs = model["scaler"].transform(Z)                       # 标准化
    return model["ridge"].predict(Zs)                       # 预测


def raw_formula(model):
    """把模型还原成"可以直接用的显式公式"，方便写论文。

    返回 (terms, intercept, centers)：
        terms      = [("√Δp*", 系数), ("P*", 系数), ...]  共 14 项（* 表示中心化后的变量）
        intercept  = 常数项（物理意义：所有自变量取训练集均值时的预测值）
        centers    = {"√Δp": 均值, "P": 均值, ...}  中心化要减掉的数
        
        scaler     = 存放的mean_(期望)、scale_(标准差)

    为什么要"还原"？
        模型是在"标准化后的基函数"上拟合的，那套系数不能直接用。
        这里做一步数学逆变换，把系数换回 √Δp、P、T、Q_in 本身。
        model is dict returned by fit_model
    """
    scale = np.asarray(model["scaler"].scale_)
    mean = np.asarray(model["scaler"].mean_)
    w = np.asarray(model["ridge"].coef_)
    b = float(model["ridge"].intercept_)

    # 标准化：   Zs = (Z - mean) / scale
    # 代回去：   y  = Σ w_i * (Z_i - mean_i)/scale_i + b
    #              = Σ (w_i/scale_i) * Z_i  +  [b - Σ w_i*mean_i/scale_i]
    coef_on_Z = w / scale
    intercept = b - float(np.sum(w * mean / scale))

    # 项的名字，例如 "√Δp P"、"P^2"
    names = ["%s*" % n for n in model["poly"].get_feature_names_out(FEATURE_LABELS)]
    terms = list(zip(names, coef_on_Z))

    centers = dict(zip(FEATURE_LABELS, np.asarray(model["center"])))
    return terms, intercept, centers
