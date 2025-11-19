# -*- coding: utf-8 -*-
"""
医学栖息地分析管道 (Medical Habitat Analysis Pipeline)

本模块集成了K-Means聚类算法进行子区域分割和PyRadiomics进行纹理特征提取。
专为2D ROI（感兴趣区域）图像块设计，用于医学影像分析。

主要功能：
1. 基于图像强度的K-Means聚类，将ROI分割为不同的栖息地（低、中、高强度区域）
2. 使用PyRadiomics提取各种纹理特征（一阶统计、GLCM、GLRLM、GLSZM、GLDM）
3. 批量处理图像并输出特征到CSV文件

"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.cluster import KMeans
from pyradiomics.radiomics import featureextractor

# --- 日志配置 ---
# 抑制冗余的库日志输出，保持应用程序日志清晰
logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """
    特征提取管道配置类
    
    使用dataclass装饰器创建配置数据类，包含所有可配置的参数。
    这样可以提高代码的可读性和可维护性。
    """
    n_clusters: int = 3  # K-Means聚类的簇数量，默认分为3个栖息地
    # 纹理计算的Bin宽度。如果输入是CT(HU)图像，建议25；如果是MRI/归一化后图像(0-1)，需调整为0.05等
    bin_width: float = 25.0 
    min_roi_pixels: int = 10  # 最小有效像素阈值，用于过滤过小的ROI区域

class HabitatExtractor:
    """
    栖息地特征提取器主类
    
    负责处理栖息地生成和特征提取的完整生命周期。
    该类封装了图像加载、栖息地分割、特征提取等核心功能。
    """

    def __init__(self, config: ExtractionConfig):
        """
        初始化栖息地提取器
        
        Args:
            config (ExtractionConfig): 特征提取配置对象
        """
        self.cfg = config
        self.extractor = self._init_pyradiomics()

    def _init_pyradiomics(self) -> featureextractor.RadiomicsFeatureExtractor:
        """
        初始化PyRadiomics特征提取器，配置2D图像处理设置
        
        Returns:
            featureextractor.RadiomicsFeatureExtractor: 配置好的PyRadiomics提取器
        """
        # PyRadiomics参数配置
        settings = {
            'binWidth': self.cfg.bin_width,  # 直方图分箱宽度
            'resampledPixelSpacing': None,  # 假设输入图像块已经具有一致的间距
            'interpolator': 'sitkBSpline',  # 插值器类型
            'force2D': True,  # 强制使用2D处理
            'force2Ddimension': 0,  # 2D处理的维度（0表示第一个维度）
            'label': 1,  # 默认标签值，将在后续动态覆盖
            'normalize': False  # 假设在此步骤之前已经完成了显式预处理
        }
        
        # 创建特征提取器并应用设置
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()  # 禁用所有特征
        
        # 启用适合与ResNet连接的纹理特征
        extractor.enableFeatureClassByName('firstorder')  # 一阶统计特征
        extractor.enableFeatureClassByName('glcm')        # 灰度共生矩阵特征
        extractor.enableFeatureClassByName('glrlm')       # 灰度游程长度矩阵特征
        extractor.enableFeatureClassByName('glszm')       # 灰度大小区域矩阵特征
        extractor.enableFeatureClassByName('gldm')        # 灰度依赖矩阵特征
        
        return extractor

    def _load_image(self, image_path: str) -> sitk.Image:
        """
        加载图像并确保为单通道float32格式
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            sitk.Image: 加载并转换后的SimpleITK图像对象
            
        Raises:
            FileNotFoundError: 当图像文件不存在时抛出
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"路径不存在: {image_path}")
            
        # 使用SimpleITK读取图像
        img = sitk.ReadImage(image_path)
        
        # 处理RGB/RGBA图像 -> 转换为灰度图
        if img.GetNumberOfComponentsPerPixel() > 1:
            img = sitk.VectorMagnitude(img)  # 计算向量幅度（RGB到灰度）
            
        # 转换为float32类型以确保数值精度
        return sitk.Cast(img, sitk.sitkFloat32)

    def _generate_habitat_mask(self, image: sitk.Image) -> Optional[sitk.Image]:
        """
        基于图像强度将ROI聚类分割为不同的栖息地
        
        使用K-Means聚类算法根据像素强度值将ROI区域分割为多个子区域（栖息地）。
        聚类结果按强度中心排序，确保标签的一致性（1=低强度，2=中强度，3=高强度）。
        
        Args:
            image (sitk.Image): 输入的2D图像对象
            
        Returns:
            Optional[sitk.Image]: 多标签掩码图像，标签含义：
                                 1: 低强度栖息地, 2: 中强度栖息地, 3: 高强度栖息地
                                 返回None表示聚类失败或ROI过小
        """
        # 将SimpleITK图像转换为numpy数组以便使用sklearn
        arr = sitk.GetArrayFromImage(image)
        
        # 定义ROI区域：假设0为背景（黑色）
        # 注意：确保你的图像块已经预处理（背景=0）
        roi_mask = arr > 1e-5  # 使用小阈值避免浮点精度问题
        valid_pixels = arr[roi_mask].reshape(-1, 1)  # 提取ROI内的有效像素

        # 检查ROI大小是否足够进行聚类
        min_required_pixels = max(self.cfg.min_roi_pixels, self.cfg.n_clusters * 2)
        if valid_pixels.shape[0] < min_required_pixels:
            LOGGER.warning(f"ROI过小，无法进行聚类。有效像素数: {valid_pixels.shape[0]}, 最小要求: {min_required_pixels}")
            return None

        try:
            # 执行K-Means聚类
            kmeans = KMeans(
                n_clusters=self.cfg.n_clusters, 
                random_state=42,  # 固定随机种子确保结果可重现
                n_init=10         # 使用不同的质心初始化运行10次，选择最佳结果
            )
            labels = kmeans.fit_predict(valid_pixels)

            # 按强度中心排序聚类结果以确保一致性
            # 0->低强度, 1->中强度, 2->高强度
            centroids = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centroids)  # 按强度值排序
            
            # 创建标签映射：旧标签 -> 新标签（从1开始）
            # 例如：如果聚类2的强度最小，则映射 2 -> 1
            label_map = {old: new + 1 for new, old in enumerate(sorted_indices)}
            
            # 将标签映射回ROI形状
            mapped_labels = np.vectorize(label_map.get)(labels)
            
            # 创建栖息地数组并填充标签
            habitat_arr = np.zeros_like(arr, dtype=np.uint8)
            habitat_arr[roi_mask] = mapped_labels
            
            # 转换回SimpleITK图像，关键是要保留空间元数据
            habitat_img = sitk.GetImageFromArray(habitat_arr)
            habitat_img.CopyInformation(image)  # 复制原始图像的空间信息
            
            return habitat_img

        except Exception as e:
            LOGGER.error(f"聚类失败: {e}")
            return None

    def process_single_patch(self, image_path: str) -> Optional[Dict[str, float]]:
        """
        处理单个图像块的主要执行单元
        
        该方法是整个管道的核心执行函数，依次执行：
        1. 图像加载和预处理
        2. 栖息地掩码生成
        3. 对每个栖息地提取特征
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            Optional[Dict[str, float]]: 特征字典，包含图像名称和所有栖息地的特征
                                      返回None表示处理失败
        """
        try:
            # 加载和预处理图像
            img_obj = self._load_image(image_path)
            
            # 生成栖息地掩码
            mask_obj = self._generate_habitat_mask(img_obj)
            
            if mask_obj is None:
                LOGGER.warning(f"无法为图像 {image_path} 生成栖息地掩码")
                return None

            # 初始化特征字典，包含图像名称
            features = {'Image_Name': os.path.basename(image_path)}
            # 栖息地标签映射
            habitat_map = {1: 'Low', 2: 'Mid', 3: 'High'}

            # 遍历所有栖息地标签，重用同一个掩码对象
            for label_id, label_name in habitat_map.items():
                try:
                    # 检查掩码中是否存在该标签，避免PyRadiomics错误
                    stats = sitk.LabelStatisticsImageFilter()
                    stats.Execute(img_obj, mask_obj)
                    if not stats.HasLabel(label_id):
                        LOGGER.debug(f"掩码中不存在标签 {label_id}，跳过")
                        continue

                    # 使用动态标签参数执行特征提取
                    # 高效：此处不创建新的SITK对象
                    result = self.extractor.execute(img_obj, mask_obj, label=label_id)
                    
                    # 提取特征并重命名列
                    for key, val in result.items():
                        if 'diagnostics' not in key:  # 跳过诊断信息
                            col_name = f"Hab_{label_name}_{key}"  # 格式：Hab_特征类型_特征名
                            features[col_name] = float(val)
                            
                except Exception as inner_e:
                    LOGGER.warning(f"标签 {label_name} ({label_id}) 的特征提取失败: {inner_e}")
                    continue

            return features

        except Exception as e:
            LOGGER.error(f"处理图像 {image_path} 时发生错误: {e}")
            return None

def main():
    """
    主函数：执行完整的栖息地特征提取管道
    
    处理流程：
    1. 配置参数初始化
    2. 创建栖息地提取器实例
    3. 扫描输入目录中的图像文件
    4. 批量处理图像并提取特征
    5. 保存结果到CSV文件
    """
    # 配置参数：请根据你的数据调整binWidth！
    # 如果是PNG图像（0-255），bin_width=25是合适的
    # 如果是标准化的浮点图像（-3.0到3.0），bin_width必须更小（例如0.1）
    config = ExtractionConfig(n_clusters=3, bin_width=25)
    pipeline = HabitatExtractor(config)

    # 数据目录路径，根据实际情况修改
    data_dir = './data/patches'
    
    # 获取图像文件列表
    if os.path.exists(data_dir):
        image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
    else:
        LOGGER.warning(f"数据目录不存在: {data_dir}")
        image_files = []
    
    results = []  # 存储所有图像的特征结果
    
    LOGGER.info(f"开始处理 {len(image_files)} 张图像...")
    
    # 批量处理图像
    for idx, fpath in enumerate(image_files):
        # 每处理100张图像输出一次进度
        if idx % 100 == 0 and idx > 0:
            LOGGER.info(f"已处理 {idx}/{len(image_files)} 张图像")
            
        # 处理单张图像
        feats = pipeline.process_single_patch(fpath)
        if feats:
            results.append(feats)

    # 保存结果
    if results:
        # 创建DataFrame并填充缺失值
        df = pd.DataFrame(results)
        df.fillna(0, inplace=True)  # 将缺失值填充为0
        
        # 输出到CSV文件
        output_path = 'habitat_features.csv'
        df.to_csv(output_path, index=False)
        LOGGER.info(f"处理完成。数据形状: {df.shape}。结果已保存到 {output_path}")
    else:
        LOGGER.warning("没有提取到任何特征。")

if __name__ == '__main__':
    main()
