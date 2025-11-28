"""命令行入口。"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import logging
from typing import Optional, Sequence

from habitat_pipeline.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    ExtractionConfig,
    load_app_config,
)
from habitat_pipeline.extractor import HabitatExtractor
from habitat_pipeline.logging_utils import setup_logging
from habitat_pipeline.pipeline import run_pipeline


def build_parser(app_config: AppConfig, config_path: Path) -> ArgumentParser:
    """创建命令行参数解析器。"""
    parser = ArgumentParser(description="医学生境分析管道")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help=f"配置文件路径 (JSON)，默认 {config_path}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=app_config.pipeline.data_dir,
        help=f"输入图像数据目录，默认 {app_config.pipeline.data_dir}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=app_config.pipeline.output_dir,
        help=f"输出 CSV 存放目录，默认 {app_config.pipeline.output_dir}",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=app_config.extraction.n_clusters,
        help=f"K-Means 聚类的簇数，默认 {app_config.extraction.n_clusters}",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=app_config.extraction.bin_width,
        help=f"PyRadiomics 灰度 bin 宽度，默认 {app_config.extraction.bin_width}",
    )
    parser.add_argument(
        "--min-roi-pixels",
        type=int,
        default=app_config.extraction.min_roi_pixels,
        help=f"ROI 最小像素数，低于该值跳过，默认 {app_config.extraction.min_roi_pixels}",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出，仅保留 ERROR 级别。",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """解析命令行参数。"""
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre_parser.parse_known_args(argv)

    app_config = load_app_config(pre_args.config)
    parser = build_parser(app_config, pre_args.config)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI 主入口。"""
    args = parse_args(argv)
    setup_logging(level=logging.INFO, quiet=args.quiet)

    config = ExtractionConfig(
        n_clusters=args.clusters,
        bin_width=args.bin_width,
        min_roi_pixels=args.min_roi_pixels,
    )
    extractor = HabitatExtractor(config)
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        extractor=extractor,
    )


if __name__ == "__main__":
    main()
