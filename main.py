# -*- coding: utf-8 -*-
"""
医学生境分析管道 (Medical Habitat Analysis Pipeline) - Version

变更点 (Mentor Refined):
1. 强制统一 Spacing 为 (1.0, 1.0)，消除 Resize 带来的物理歧义。
2. 优化 LabelStatistics 性能，避免循环内重复计算。
3. 增强了数据类型的健壮性。
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.cluster import KMeans
from pyradiomics.radiomics import featureextractor

# --- 日志配置 ---
logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """特征提取管道配置类"""
    n_clusters: int = 3
    bin_width: float = 25.0 
    min_roi_pixels: int = 10

class HabitatExtractor:
    """栖息地特征提取器主类"""

    def __init__(self, config: ExtractionConfig):
        self.cfg = config
        self.extractor = self._init_pyradiomics()

    def _init_pyradiomics(self) -> featureextractor.RadiomicsFeatureExtractor:
        """初始化 PyRadiomics"""
        settings = {
            'binWidth': self.cfg.bin_width,
            'resampledPixelSpacing': None,  # 显式禁用重采样，完全信任输入的 (1,1) 间距
            'interpolator': 'sitkBSpline',
            'force2D': True,
            'force2Ddimension': 0,
            'label': 1,
            'normalize': False
        }
        
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        # 启用适合生境分析的纹理特征
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('gldm')
        
        return extractor

    def _load_image(self, image_path: Path) -> sitk.Image:
        """加载图像并标准化元数据"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"路径不存在: {image_path}")
            
        img = sitk.ReadImage(image_path)
        
        # RGB -> Gray
        if img.GetNumberOfComponentsPerPixel() > 1:
            img = sitk.VectorMagnitude(img)
            
        # 转 float64
        img = sitk.Cast(img, sitk.sitkFloat64)
        
        # --- 【导师核心修正】强制归一化空间元数据 ---
        # 针对 Resize 后的 Patch，物理尺寸已无意义。
        # 强制设为单位像素间距，确保所有样本基于"像素距离"计算纹理，保证可比性。
        img.SetSpacing((1.0, 1.0))
        img.SetOrigin((0.0, 0.0))
        img.SetDirection((1.0, 0.0, 0.0, 1.0))
        
        return img

    def _generate_habitat_mask(self, image: sitk.Image) -> Optional[sitk.Image]:
        """K-Means 聚类生成生境掩膜"""
        arr = sitk.GetArrayFromImage(image)
        
        # 提取 ROI (假设背景接近 0)
        roi_mask = arr > 1e-5 
        valid_pixels = arr[roi_mask].reshape(-1, 1)

        # 样本量检查
        min_required_pixels = max(self.cfg.min_roi_pixels, self.cfg.n_clusters * 5) # 稍微提高安全阈值
        if valid_pixels.shape[0] < min_required_pixels:
            LOGGER.warning(f"ROI过小 (n={valid_pixels.shape[0]})，跳过。")
            return None

        try:
            # K-Means
            kmeans = KMeans(
                n_clusters=self.cfg.n_clusters, 
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(valid_pixels)

            # 强度排序: 保证 Label 1=Low, 2=Mid, 3=High
            centroids = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centroids)
            label_map = {old: new + 1 for new, old in enumerate(sorted_indices)}
            
            mapped_labels = np.vectorize(label_map.get)(labels)
            
            # 重构掩膜
            habitat_arr = np.zeros_like(arr, dtype=np.uint8)
            habitat_arr[roi_mask] = mapped_labels
            
            habitat_img = sitk.GetImageFromArray(habitat_arr)
            habitat_img.CopyInformation(image) # 继承修正后的 (1,1) Spacing
            
            return habitat_img

        except Exception as e:
            LOGGER.error(f"聚类异常: {e}")
            return None

    def process_single_patch(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """处理单张图像"""
        try:
            img_obj = self._load_image(image_path)
            mask_obj = self._generate_habitat_mask(img_obj)
            
            if mask_obj is None:
                return None

            features = {'Image_Name': os.path.basename(image_path)}
            habitat_map = {1: 'Low', 2: 'Mid', 3: 'High'}

            # --- 【性能修正】统计只需做一次 ---
            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(img_obj, mask_obj)

            for label_id, label_name in habitat_map.items():
                try:
                    # 检查 Label 是否存在
                    if not stats.HasLabel(label_id):
                        continue

                    # 提取特征
                    result = self.extractor.execute(img_obj, mask_obj, label=label_id)
                    
                    for key, val in result.items():
                        if 'diagnostics' not in key:
                            # 统一前缀方便后续分析
                            col_name = f"Hab_{label_name}_{key}"
                            features[col_name] = float(val)
                            
                except Exception as inner_e:
                    LOGGER.warning(f"Label {label_name} 提取失败: {inner_e}")
                    continue

            return features

        except Exception as e:
            LOGGER.error(f"处理失败 {image_path}: {e}")
            return None

def main():
    # 你的数据如果是 0-255 的 PNG，binWidth=25 是完美的
    config = ExtractionConfig(n_clusters=3, bin_width=25)
    pipeline = HabitatExtractor(config)

    data_dir = Path('/home/wwt/data/raw/liver/Grade')

    if os.path.exists(data_dir):
        image_files: List[Path] = []
        for image_path in data_dir.rglob('*.png'):
            image_files.append(Path(image_path))
    else:
        LOGGER.warning(f"目录不存在: {data_dir}")
        return # 直接返回，不要继续
    if not image_files:
        LOGGER.warning("未找到 PNG 图像。")
        return
    
    results = []
    LOGGER.info(f"开始处理 {len(image_files)} 张图像...")
    
    for idx, fpath in enumerate(image_files):
        if idx % 100 == 0 and idx > 0:
            LOGGER.info(f"进度: {idx}/{len(image_files)}")
            
        feats = pipeline.process_single_patch(fpath)
        if feats:
            results.append(feats)

    if results:
        df = pd.DataFrame(results)
        df.fillna(0, inplace=True)
        output_path = '/home/wwt/data/outputs/habitat/habitat_features.csv'
        df.to_csv(output_path, index=False)
        LOGGER.info(f"完成！特征已保存至 {output_path}")
    else:
        LOGGER.warning("结果为空。")

if __name__ == '__main__':
    main()
