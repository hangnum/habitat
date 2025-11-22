"""命令行入口。"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import logging
from typing import Optional, Sequence

from habitat_pipeline.config import ExtractionConfig
from habitat_pipeline.extractor import HabitatExtractor
from habitat_pipeline.logging_utils import setup_logging
from habitat_pipeline.pipeline import run_pipeline


def build_parser() -> ArgumentParser:
    """创建命令行参数解析器。"""
    parser = ArgumentParser(description="医学生境分析管道")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/wwt/data/raw/liver/Grade"),
        help="输入图像数据目录，默认 /home/wwt/data/raw/liver/Grade",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/wwt/data/outputs/habitat/"),
        help="输出 CSV 存放目录，默认 /home/wwt/data/outputs/habitat/",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="K-Means 聚类的簇数，默认 3",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=25.0,
        help="PyRadiomics 灰度 bin 宽度，默认 25",
    )
    parser.add_argument(
        "--min-roi-pixels",
        type=int,
        default=10,
        help="ROI 最小像素数，低于该值跳过，默认 10",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出，仅保留 ERROR 级别。",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """解析命令行参数。"""
    parser = build_parser()
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

