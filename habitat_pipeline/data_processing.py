"""数据处理模块：特征加载、聚合与预处理。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """筛选出可用的数值型特征列。"""
    exclude = {"Image_Name", "Patient_ID", "Patient_Label", "Hospital", "Modality"}
    return [
        col
        for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def load_feature_tables(feature_dir: Path) -> pd.DataFrame:
    """读取提取阶段的 CSV 特征并合并。"""
    csv_files = sorted(feature_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {feature_dir} 未找到特征 CSV。")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        feature_cols = numeric_feature_columns(df)
        # 强制转换为数值，无法转换的置为 NaN（不要在这里填 0）
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
        LOGGER.info("载入特征文件 %s，包含数值特征列: %s", path.name, len(feature_cols))

    merged = pd.concat(frames, ignore_index=True)
    original_len = len(merged)

    total_cells = merged.shape[0] * max(merged.shape[1], 1)
    total_na = int(merged.isna().sum().sum())
    if total_na > 0 and total_cells > 0:
        na_ratio = 100.0 * total_na / total_cells
        LOGGER.info(
            "已合并所有特征文件，样本总数: %s；整体缺失率约为 %.2f%%，将在建模 Pipeline 中使用中位数插补。",
            original_len,
            na_ratio,
        )
    else:
        LOGGER.info("已合并所有特征文件，样本总数: %s；无缺失值。", original_len)

    return merged


def mean_pool_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """按医院/病人/模态做平均池化，获得患者级特征。"""
    group_keys = ["Hospital", "Patient_ID", "Patient_Label", "Modality"]
    feature_cols = numeric_feature_columns(df)

    # 聚合前检查
    if not feature_cols:
        raise ValueError("未检测到有效的数值特征列。")

    # groupby.mean 默认会跳过 NaN
    pooled = df.groupby(group_keys)[feature_cols].mean().reset_index()
    LOGGER.info("池化完成，患者-模态组合数: %s", len(pooled))
    return pooled


def filter_modalities(pooled: pd.DataFrame, allowed: Sequence[str]) -> pd.DataFrame:
    """仅保留指定模态的数据。"""
    allowed_set = set(allowed)
    filtered = pooled[pooled["Modality"].isin(allowed_set)].copy()
    dropped = pooled.shape[0] - filtered.shape[0]

    if dropped > 0:
        LOGGER.info("已过滤非指定模态，丢弃样本数: %s", dropped)
    if filtered.empty:
        raise ValueError(f"过滤后无可用数据，期望模态: {sorted(allowed_set)}")

    return filtered


def drop_incomplete_modalities(
    pooled: pd.DataFrame, required_modalities: Sequence[str]
) -> pd.DataFrame:
    """剔除缺失任一指定模态的患者。"""
    base_keys = ["Hospital", "Patient_ID", "Patient_Label"]
    required_set = set(required_modalities)

    # 计算每个患者拥有的模态集合
    modality_sets = (
        pooled.groupby(base_keys)["Modality"]
        .agg(lambda s: frozenset(s))
        .reset_index(name="mods")
    )

    # 筛选拥有所有必需模态的患者 ID
    complete_keys = modality_sets[
        modality_sets["mods"].apply(lambda mods: required_set.issubset(mods))
    ][base_keys]

    n_before = modality_sets.shape[0]
    n_after = complete_keys.shape[0]
    dropped = n_before - n_after

    if dropped > 0:
        LOGGER.info("剔除缺失模态的患者数: %s (剩余有效患者: %s)", dropped, n_after)
    if n_after == 0:
        raise ValueError(f"所有患者均缺少所需模态 {sorted(required_set)}")

    # 仅保留完整患者的数据
    return pooled.merge(complete_keys, on=base_keys, how="inner")


def merge_modalities(pooled: pd.DataFrame, allowed_modalities: Sequence[str]) -> pd.DataFrame:
    """将不同模态的患者级特征横向拼接。"""
    base_keys = ["Hospital", "Patient_ID", "Patient_Label"]
    feature_cols = numeric_feature_columns(pooled)

    modality_frames: List[pd.DataFrame] = []
    for modality, part in pooled.groupby("Modality"):
        if modality not in allowed_modalities:
            continue
        # 重命名特征列，防止重名
        renamed = part.rename(
            columns={col: f"{col}_{modality}" for col in feature_cols},
        ).drop(columns=["Modality"])
        modality_frames.append(renamed)

    if not modality_frames:
        raise ValueError("未找到待拼接的模态数据。")

    merged = modality_frames[0]
    for frame in modality_frames[1:]:
        # 使用 inner 连接，因为此前已执行 drop_incomplete_modalities，inner 更安全
        merged = pd.merge(merged, frame, on=base_keys, how="inner")

    LOGGER.info(
        "模态拼接完成，最终患者数: %s，特征总维数: %s",
        len(merged),
        len(merged.columns) - len(base_keys),
    )
    return merged


def split_cohorts(
    df: pd.DataFrame,
    internal_hospital: str,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """划分训练、内验和外验集。"""
    internal = df[df["Hospital"] == internal_hospital].reset_index(drop=True)
    external = df[df["Hospital"] != internal_hospital].reset_index(drop=True)

    if internal.empty:
        raise ValueError(f"未找到医院 {internal_hospital} 的样本。")

    labels = internal["Patient_Label"]
    stratify_col = labels if labels.nunique() > 1 else None

    if len(internal) < 5:
        LOGGER.warning("内部医院样本过少，无法划分验证集。")
        train_df, val_df = internal, internal.iloc[0:0]
    else:
        train_df, val_df = train_test_split(
            internal,
            test_size=val_size,
            random_state=seed,
            stratify=stratify_col,
        )

    LOGGER.info(
        "数据划分: Train=%s, Val=%s, External=%s",
        len(train_df),
        len(val_df),
        len(external),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), external