"""面向批量处理的管线调度。"""

from pathlib import Path
from typing import Any, Dict, List

import logging

import pandas as pd
from tqdm import tqdm

from habitat_pipeline.dataset import structure_data_by_hospital
from habitat_pipeline.extractor import HabitatExtractor

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    extractor: HabitatExtractor,
) -> None:
    """执行完整的医院-模态生境特征提取。

    Args:
        data_dir: 输入数据的根目录。
        output_dir: 输出 CSV 的存储目录。
        extractor: 已配置好的栖息地特征提取器。
    """
    if not data_dir.exists():
        LOGGER.error("数据目录不存在: %s", data_dir)
        return

    image_files = list(data_dir.rglob("*.png"))
    if not image_files:
        LOGGER.warning("在 %s 中未找到 PNG 图像。", data_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("找到了 %s 张图像，开始构建医院-模态-病人映射...", len(image_files))
    hospital_data = structure_data_by_hospital(image_files, data_dir)
    LOGGER.info("数据结构构建完成。")

    total_groups = sum(len(modalities) for modalities in hospital_data.values())
    processed_groups = 0

    LOGGER.info("开始按 医院 -> 模态 的顺序处理图像...")
    for hospital, modality_data in hospital_data.items():
        for modality, patients in modality_data.items():
            processed_groups += 1
            LOGGER.info("--- [进度 %s/%s] ---", processed_groups, total_groups)
            LOGGER.info("正在处理 医院: [%s], 模态: [%s]", hospital, modality)

            modality_results: List[Dict[str, Any]] = []
            for patient_id, patient_info in tqdm(
                patients.items(),
                desc=f"  [{hospital}-{modality}] 处理病人",
                unit="病人",
                leave=False,
            ):
                for fpath in patient_info["image_paths"]:
                    feats = extractor.process_single_patch(fpath)
                    if feats:
                        feats.update(
                            {
                                "Patient_ID": patient_id,
                                "Patient_Label": patient_info["label"],
                                "Hospital": hospital,
                                "Modality": modality,
                            }
                        )
                        modality_results.append(feats)

            if modality_results:
                df = pd.DataFrame(modality_results)
                df.fillna(0, inplace=True)

                output_path = output_dir / f"{hospital}_{modality}.csv"
                df.to_csv(output_path, index=False)
                LOGGER.info("处理完成, 特征已保存至: %s", output_path)
            else:
                LOGGER.warning("医院 [%s], 模态 [%s] 未产生任何结果。", hospital, modality)

    LOGGER.info("所有医院及模态均已处理完毕！")

