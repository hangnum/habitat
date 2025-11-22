"""数据组织工具。"""

from pathlib import Path
from typing import Any, Dict, List

import logging

LOGGER = logging.getLogger(__name__)


def structure_data_by_hospital(
    image_files: List[Path],
    data_dir: Path,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """按医院与模态构建层级字典。

    期望的路径结构: ``<data_dir>/gradeX/<patient>/<modality>/<file>.png``，其中 gradeX 可转为整数标签。

    Args:
        image_files: 待处理的 PNG 路径列表。
        data_dir: 数据根目录，用于解析相对路径。

    Returns:
        Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]: 三层嵌套的病人映射。
    """
    hospital_data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    for path in image_files:
        try:
            relative_parts = path.relative_to(data_dir).parts

            hospital = relative_parts[0]
            modality = path.parent.name
            patient_id = path.parent.parent.name
            label_str = path.parent.parent.parent.name
            label = int(label_str.replace("grade", ""))

            modality_dict = hospital_data.setdefault(hospital, {})
            patient_dict = modality_dict.setdefault(modality, {})
            patient_info = patient_dict.setdefault(
                patient_id, {"label": label, "image_paths": []}
            )
            patient_info["image_paths"].append(path)
        except (IndexError, ValueError) as exc:
            LOGGER.warning("解析路径失败 %s，跳过。错误: %s", path, exc)
            continue

    return hospital_data
