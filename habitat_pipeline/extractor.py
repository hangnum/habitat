"""影像载入与栖息地特征提取逻辑。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
from pyradiomics.radiomics import featureextractor

from habitat_pipeline.config import ExtractionConfig

LOGGER = logging.getLogger(__name__)


class HabitatExtractor:
    """栖息地特征提取器。

    负责影像的标准化、K-Means 生境分割以及基于 PyRadiomics 的纹理特征计算。

    Attributes:
        cfg: 特征提取配置。
        extractor: PyRadiomics 的特征提取实例。
    """

    def __init__(self, config: ExtractionConfig):
        self.cfg = config
        self.extractor = self._init_pyradiomics()

    def _init_pyradiomics(self) -> featureextractor.RadiomicsFeatureExtractor:
        """初始化 PyRadiomics 提取器。

        Returns:
            featureextractor.RadiomicsFeatureExtractor: 已配置的提取器。
        """
        settings = {
            "binWidth": self.cfg.bin_width,
            "resampledPixelSpacing": None,  # 显式禁用重采样，完全信任输入的 (1,1) 间距
            "interpolator": "sitkBSpline",
            "force2D": True,
            "force2Ddimension": 0,
            "label": 1,
            "normalize": False,
        }

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("firstorder")
        extractor.enableFeatureClassByName("glcm")
        extractor.enableFeatureClassByName("glrlm")
        extractor.enableFeatureClassByName("glszm")
        extractor.enableFeatureClassByName("gldm")

        return extractor

    def _load_image(self, image_path: Path) -> sitk.Image:
        """加载并标准化影像。

        将彩色图转为灰度，强制设置单位空间元数据，确保纹理计算基于像素距离。

        Args:
            image_path: 图像路径。

        Returns:
            sitk.Image: 标准化后的影像。

        Raises:
            FileNotFoundError: 输入路径不存在。
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"路径不存在: {image_path}")

        img = sitk.ReadImage(image_path)

        if img.GetNumberOfComponentsPerPixel() > 1:
            img = sitk.VectorMagnitude(img)

        img = sitk.Cast(img, sitk.sitkFloat64)
        img.SetSpacing((1.0, 1.0))
        img.SetOrigin((0.0, 0.0))
        img.SetDirection((1.0, 0.0, 0.0, 1.0))

        return img

    def _generate_habitat_mask(self, image: sitk.Image) -> Optional[sitk.Image]:
        """基于 K-Means 的生境掩膜生成。

        Args:
            image: 已标准化的影像。

        Returns:
            Optional[sitk.Image]: 生境掩膜，若 ROI 过小或聚类失败返回 None。
        """
        arr = sitk.GetArrayFromImage(image)
        roi_mask = arr > 1e-5
        valid_pixels = arr[roi_mask].reshape(-1, 1)

        min_required_pixels = max(self.cfg.min_roi_pixels, self.cfg.n_clusters * 5)
        if valid_pixels.shape[0] < min_required_pixels:
            LOGGER.warning("ROI 过小 (n=%s)，跳过。", valid_pixels.shape[0])
            return None

        try:
            kmeans = KMeans(
                n_clusters=self.cfg.n_clusters,
                random_state=42,
                n_init=10,
            )
            labels = kmeans.fit_predict(valid_pixels)

            centroids = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centroids)
            label_map = {old: new + 1 for new, old in enumerate(sorted_indices)}
            mapped_labels = np.vectorize(label_map.get)(labels)

            habitat_arr = np.zeros_like(arr, dtype=np.uint8)
            habitat_arr[roi_mask] = mapped_labels

            habitat_img = sitk.GetImageFromArray(habitat_arr)
            habitat_img.CopyInformation(image)

            return habitat_img
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("聚类异常: %s", exc)
            return None

    def process_single_patch(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """处理单张图像，返回特征字典。

        Args:
            image_path: 图像路径。

        Returns:
            Optional[Dict[str, Any]]: 包含图像名及多生境纹理特征的键值对。
        """
        try:
            img_obj = self._load_image(image_path)
            mask_obj = self._generate_habitat_mask(img_obj)

            if mask_obj is None:
                return None

            features: Dict[str, Any] = {"Image_Name": os.path.basename(image_path)}
            habitat_map = {1: "Low", 2: "Mid", 3: "High"}

            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(img_obj, mask_obj)

            for label_id, label_name in habitat_map.items():
                if not stats.HasLabel(label_id):
                    continue

                try:
                    result = self.extractor.execute(img_obj, mask_obj, label=label_id)
                    for key, val in result.items():
                        if "diagnostics" not in key:
                            col_name = f"Hab_{label_name}_{key}"
                            features[col_name] = float(val)
                except Exception as inner_exc:  # noqa: BLE001
                    LOGGER.warning("Label %s 提取失败: %s", label_name, inner_exc)
                    continue

            return features
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("处理失败 %s: %s", image_path, exc)
            return None

