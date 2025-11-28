"""配置定义与加载模块。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# 默认配置文件路径，位于仓库根目录下
DEFAULT_CONFIG_PATH = Path("config/defaults.json")


@dataclass
class ExtractionConfig:
    """生境特征提取配置。"""

    n_clusters: int = 3
    bin_width: float = 25.0
    min_roi_pixels: int = 10


@dataclass
class PipelineConfig:
    """特征提取管线的输入输出配置。"""

    data_dir: Path = Path("./data/raw/liver/Grade")
    output_dir: Path = Path("./outputs/habitat")


@dataclass
class TrainingConfig:
    """特征聚合与模型训练配置。"""

    input_dir: Path = Path("./outputs/habitat")
    output_dir: Path = Path("./feature")
    internal_hospital: str = "JM"
    val_size: float = 0.2
    seed: int = 42
    modalities: list[str] = field(default_factory=lambda: ["A", "P"])
    log_level: str = "INFO"


@dataclass
class AppConfig:
    """顶层配置对象，聚合了各子模块配置。"""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _apply_overrides(target: Any, overrides: Dict[str, Any]) -> None:
    """将字典中的配置项覆盖到 dataclass 实例上。"""
    if not isinstance(overrides, dict):
        return

    for field_def in fields(target):
        if field_def.name not in overrides:
            continue

        incoming = overrides[field_def.name]
        current_val = getattr(target, field_def.name)

        if is_dataclass(current_val):
            _apply_overrides(current_val, incoming)
            continue

        if isinstance(current_val, Path):
            setattr(target, field_def.name, Path(incoming))
            continue

        setattr(target, field_def.name, incoming)


def load_app_config(config_path: Optional[Path] = None) -> AppConfig:
    """从 JSON 配置文件加载配置，文件缺失时回退到默认值。

    Args:
        config_path: 配置文件路径，默认读取 ``config/defaults.json``。

    Returns:
        AppConfig: 已合并配置文件的配置对象。
    """
    cfg = AppConfig()
    path = Path(config_path or DEFAULT_CONFIG_PATH)

    if not path.exists():
        LOGGER.info("未找到配置文件 %s，使用默认值。", path)
        return cfg

    try:
        with path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
    except json.JSONDecodeError as exc:  # noqa: B902
        raise ValueError(f"配置文件解析失败: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"配置文件格式错误，应为 JSON 对象: {path}")

    for section in ("pipeline", "extraction", "training"):
        if section in raw:
            _apply_overrides(getattr(cfg, section), raw[section])

    return cfg
