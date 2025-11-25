"""基于已提取特征的患者级聚合、特征筛选与逻辑回归超参数调优（模块化版本）。"""

from __future__ import annotations

import json
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Sequence

# 忽略 sklearn 在极小样本下的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .data_processing import (
    load_feature_tables,
    mean_pool_by_patient,
    filter_modalities,
    drop_incomplete_modalities,
    merge_modalities,
    split_cohorts,
)
from .model_training import train_and_tune_model
from .utils import setup_logging, set_random_seed, save_model_coefficients


def build_parser() -> ArgumentParser:
    """构建命令行参数。"""
    parser = ArgumentParser(description="医学影像特征聚合与 Robust LR 训练（模块化版本）")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/wwt/data/outputs/habitat"),
        help="提取特征 CSV 所在目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./feature"),
        help="患者级特征与模型输出目录。",
    )
    parser.add_argument(
        "--internal-hospital",
        type=str,
        default="JM",
        help="划分训练+内验的医院名，其他医院作为外验。",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="内验证集占比（仅作用于 internal hospital）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于数据划分和模型训练。",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["A", "P"],
        help="仅对这些模态进行横向拼接。",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别。",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="仅生成患者级特征与数据划分，不训练模型。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """主程序入口。"""
    args = build_parser().parse_args(argv)

    # 设置日志和随机种子
    setup_logging(args.log_level)
    set_random_seed(args.seed)

    # 1. 准备目录
    feature_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 数据加载与预处理
    raw_df = load_feature_tables(feature_dir)
    pooled_df = mean_pool_by_patient(raw_df)

    # 3. 模态过滤与完整性检查
    pooled_df = filter_modalities(pooled_df, args.modalities)
    pooled_df = drop_incomplete_modalities(pooled_df, args.modalities)
    pooled_df.to_csv(output_dir / "patient_modality_mean.csv", index=False)

    # 4. 模态拼接
    merged_df = merge_modalities(pooled_df, args.modalities)
    merged_df.to_csv(output_dir / "patient_features_merged.csv", index=False)

    # 5. 数据集划分
    train_df, val_df, external_df = split_cohorts(
        merged_df,
        internal_hospital=args.internal_hospital,
        val_size=args.val_size,
        seed=args.seed,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    external_df.to_csv(output_dir / "external.csv", index=False)

    if args.no_train:
        print("跳过训练流程。")
        return

    # 6. 模型训练与超参数搜索
    model, metrics = train_and_tune_model(train_df, val_df, external_df, args.seed)

    # 7. 保存结果
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    feature_cols = [
        col
        for col in train_df.columns
        if col not in {"Hospital", "Patient_ID", "Patient_Label"}
    ]
    save_model_coefficients(model, feature_cols, output_dir / "coefficients.csv")

    print(f"训练完成，结果已保存到 {output_dir}")
    print(f"最佳 CV AUC: {metrics['cv_best_auc']:.4f}")
    print(f"外部验证 AUC: {metrics['external']['auc'] or 'N/A'}")


if __name__ == "__main__":
    main()