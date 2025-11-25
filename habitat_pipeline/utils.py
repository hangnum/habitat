"""工具函数模块：评估、保存系数等通用功能。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

LOGGER = logging.getLogger(__name__)


def evaluate_model(
    name: str,
    model: Any,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Optional[float]]:
    """计算模型在给定数据集上的评估指标。"""
    if df.empty:
        return {"n": 0, "accuracy": None, "f1": None, "auc": None}

    X = df[feature_cols]
    y = df["Patient_Label"]

    preds = model.predict(X)
    probas = model.predict_proba(X)

    metrics: Dict[str, Optional[float]] = {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, average="weighted")),
    }

    try:
        if probas.shape[1] == 2:
            auc = roc_auc_score(y, probas[:, 1])
        else:
            auc = roc_auc_score(y, probas, multi_class="ovr")
        metrics["auc"] = float(auc)
    except ValueError:
        metrics["auc"] = None
        LOGGER.warning("%s 集类别单一，无法计算 AUC。", name)

    LOGGER.info(
        "[%-8s] N=%-4d Acc=%.4f F1=%.4f AUC=%s",
        name,
        metrics["n"],
        metrics["accuracy"] or 0.0,
        metrics["f1"] or 0.0,
        metrics["auc"] if metrics["auc"] is not None else "N/A",
    )
    return metrics


def save_model_coefficients(model: Any, feature_cols: List[str], output_path: Path) -> None:
    """保存模型系数，处理 Pipeline 中的特征选择。"""
    selector = model.named_steps["selector"]
    clf = model.named_steps["clf"]

    # 获取被选中的特征名称
    selected_mask = selector.get_support()
    selected_features = np.array(feature_cols)[selected_mask]

    coefs = clf.coef_
    classes = clf.classes_

    if coefs.shape[0] == 1:
        # 二分类：coefs[0] 为正类 (classes[1]) 相对于负类 (classes[0]) 的系数
        target_class = int(classes[1]) if len(classes) > 1 else int(classes[0])
        coef_df = pd.DataFrame(
            {
                "Feature": selected_features,
                "Coefficient": coefs[0],
                "TargetClass": np.repeat(target_class, len(selected_features)),
            }
        )
    else:
        # 多分类：每个类一列系数
        coef_df = pd.DataFrame(coefs.T, columns=[f"class_{c}" for c in classes])
        coef_df.insert(0, "Feature", selected_features)

    coef_df.to_csv(output_path, index=False)
    LOGGER.info("模型系数已保存 (仅保留筛选后特征): %s", output_path)


def setup_logging(level: str = "INFO") -> None:
    """配置日志设置。"""
    import logging

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_random_seed(seed: int) -> None:
    """设置随机种子以确保结果可重现。"""
    import numpy as np

    np.random.seed(seed)