"""模型训练模块：逻辑回归训练、评估与超参数调优。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from .utils import evaluate_model, save_model_coefficients

LOGGER = logging.getLogger(__name__)


def build_training_pipeline() -> Pipeline:
    """构建训练用 Pipeline，包含预处理、特征选择和分类器。"""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=20)),  # k 仅作初始化
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def get_hyperparameter_grid() -> Dict[str, List[Any]]:
    """获取超参数搜索网格。"""
    return {
        "selector__k": [10, 20, 30, 50],
        "clf__C": [0.01, 0.1, 0.5, 1.0, 2.0],
    }


def train_and_tune_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    external_df: pd.DataFrame,
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    """
    改进版训练流程:
    1. 使用 SimpleImputer 在 Pipeline 中进行插补；
    2. 使用 RobustScaler 抗离群点；
    3. 使用单变量 SelectKBest 做温和特征筛选；
    4. 下游使用 L2 Logistic Regression，并对 C 做超参搜索。
    """
    feature_cols = [
        col
        for col in train_df.columns
        if col not in {"Hospital", "Patient_ID", "Patient_Label"}
    ]
    X_train = train_df[feature_cols]
    y_train = train_df["Patient_Label"]

    # 构建 Pipeline
    pipeline = build_training_pipeline()

    # 设置分类器的随机种子
    pipeline.named_steps["clf"].set_params(random_state=seed)

    # 超参数网格
    param_grid = get_hyperparameter_grid()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    LOGGER.info("开始 GridSearchCV (Imputer + RobustScaler + SelectKBest + Logistic)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    selector = best_model.named_steps["selector"]
    final_features_n = int(selector.get_support().sum())
    LOGGER.info("最佳参数: %s", grid_search.best_params_)
    LOGGER.info("最终保留特征数: %s / %s", final_features_n, len(feature_cols))
    LOGGER.info("最佳 CV 平均 AUC: %.4f", grid_search.best_score_)

    metrics = {
        "train": evaluate_model("train", best_model, train_df, feature_cols),
        "val": evaluate_model("val", best_model, val_df, feature_cols),
        "external": evaluate_model("external", best_model, external_df, feature_cols),
        "best_params": grid_search.best_params_,
        "n_features": final_features_n,
        "cv_best_auc": float(grid_search.best_score_),
    }

    return best_model, metrics