# -*- coding: utf-8 -*-
"""
医学影像K-Means聚类与PyRadiomics特征提取流程。

该脚本实现了一个自动化流程，用于：
1. 加载NIfTI, DICOM, NRRD等格式的医学影像。
2. 使用K-Means算法基于像素/体素强度进行无监督分割。
3. 对分割结果进行后处理，提取空间上连续的区域。
4. 对每个连续区域，使用PyRadiomics库提取影像组学特征。
5. 输出特征提取结果的摘要。

遵循Google Python项目规范。
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，便于在无GUI环境下运行
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from pyradiomics.radiomics import featureextractor
from sklearn.cluster import KMeans

# 配置日志记录器
LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    存储流程所需的所有配置参数。

    属性:
      image_path: 输入影像文件的路径。
      image_format: 影像格式，'auto'表示自动检测。
      n_clusters: K-Means算法的聚类数量。
      random_state: K-Means算法的随机种子，用于保证结果可复现。
      slice_axis: 对于3D影像，选择哪个轴进行切片可视化。
      slice_index: 可视化切片的索引，None表示自动选择中间层。
      plot_path: 聚类结果可视化图片的保存路径，None表示不保存。
      max_voxels_per_cluster: 单个聚类允许的最大体素数，用于防止内存溢出。
      num_workers: 特征提取时使用的并发工作线程数。
    """
    image_path: str
    image_format: str = 'auto'
    n_clusters: int = 3
    random_state: int = 0
    slice_axis: int = -1
    slice_index: Optional[int] = None
    plot_path: Optional[str] = 'clustering_result.png'
    max_voxels_per_cluster: int = 1_000_000
    num_workers: int = 1


def load_medical_image(image_path: str,
                       format_type: str = 'auto') -> Tuple[np.ndarray, sitk.Image]:
    """
    加载医学影像文件并返回其NumPy数组和SimpleITK图像对象。

    该函数自动处理多通道图像，将其转换为单通道灰度图像。

    参数:
      image_path: 影像文件的路径。
      format_type: 影像格式，'auto'将根据文件扩展名自动推断。

    返回:
      一个元组，包含：
      - image_data (np.ndarray): 影像的NumPy数组表示。
      - image_obj (sitk.Image): 影像的SimpleITK对象，保留了元数据。

    异常:
      ValueError: 如果指定的格式类型不受支持。
      RuntimeError: 如果SimpleITK无法读取文件。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"影像文件不存在: {image_path}")

    # SimpleITK 已能处理 NIfTI, NRRD, DICOM, MHA 等多种格式，无需特殊处理。
    LOGGER.info("使用 SimpleITK 加载影像: %s", image_path)
    image_obj = sitk.ReadImage(image_path)

    # 检查并处理多通道（如彩色）影像
    if image_obj.GetNumberOfComponentsPerPixel() > 1:
        LOGGER.info("检测到多通道影像 (%s 通道)，将其转换为单通道灰度强度",
                    image_obj.GetNumberOfComponentsPerPixel())
        # VectorMagnitude 将多通道向量转换为标量大小，适用于彩色转灰度
        image_obj = sitk.VectorMagnitude(image_obj)
        # 确保转换后的图像类型为浮点型以便后续处理
        image_obj = sitk.Cast(image_obj, sitk.sitkFloat32)

    image_data = sitk.GetArrayFromImage(image_obj)
    return image_data, image_obj


def perform_kmeans_clustering(image_data: np.ndarray, n_clusters: int,
                            random_state: int = 0) -> np.ndarray:
    """
    对影像数据执行K-Means聚类。

    该聚类仅基于像素强度，不考虑空间信息。

    参数:
      image_data: 包含影像像素/体素强度的NumPy数组。
      n_clusters: 要形成的聚类数量。
      random_state: 随机数生成器的种子。

    返回:
      一个与image_data形状相同的NumPy数组，其中每个元素是其对应的聚类标签。
    """
    LOGGER.info("执行 K-Means 聚类 (clusters=%s, random_state=%s)", n_clusters,
                random_state)
    # 将多维数组展平为一维向量以进行聚类
    image_flattened = image_data.flatten().reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(image_flattened)

    LOGGER.info("K-Means 聚类完成")
    return labels.reshape(image_data.shape)


def _postprocess_segmentation_mask(
        raw_labels: np.ndarray) -> np.ndarray:
    """
    对原始分割掩码进行后处理，提取每个类别的最大连通组件。

    此步骤至关重要，因为它确保了用于特征提取的每个区域在空间上是
    连续的，从而使形态学特征（如面积、周长）具有意义。

    参数:
      raw_labels: K-Means生成的原始标签数组。

    返回:
      一个处理后的标签数组，其中每个标签对应一个单一、连续的区域。
      标签ID将从1开始重新编号。
    """
    LOGGER.info("后处理分割掩码：提取每个聚类的最大连通组件。")
    processed_labels = np.zeros_like(raw_labels, dtype=np.int32)
    next_label_id = 1
    
    unique_raw_labels = np.unique(raw_labels)

    for label in unique_raw_labels:
        # 为当前聚类创建一个二值掩码
        binary_mask = (raw_labels == label).astype(np.uint8)
        
        # 使用SimpleITK寻找所有连通的区域
        mask_sitk = sitk.GetImageFromArray(binary_mask)
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled_mask_sitk = cc_filter.Execute(mask_sitk)
        
        num_components = cc_filter.GetObjectCount()
        if num_components == 0:
            continue
        
        # 仅保留最大的连通区域
        relabel_filter = sitk.RelabelComponentImageFilter()
        relabel_filter.SortByObjectSizeOn()
        largest_component_sitk = relabel_filter.Execute(labeled_mask_sitk)
        
        # 将最大区域（现在标签为1）添加到最终的掩码中
        largest_component_np = sitk.GetArrayFromImage(largest_component_sitk)
        processed_labels[largest_component_np == 1] = next_label_id
        next_label_id += 1
    
    LOGGER.info("掩码后处理完成，生成了 %s 个连续区域。", next_label_id - 1)
    return processed_labels


def plot_cluster_slice(labels_reshaped: np.ndarray,
                       output_path: Optional[str],
                       axis: int = -1,
                       slice_index: Optional[int] = None):
    """
    将聚类结果的指定切片可视化并保存为图像文件。

    如果输入是2D图像，则直接保存整个图像。

    参数:
      labels_reshaped: 聚类标签数组。
      output_path: 图像文件的保存路径。如果为None，则不执行任何操作。
      axis: 对于3D数据，选择切片的轴。
      slice_index: 切片的索引。如果为None，则自动选择中间层。
    """
    if not output_path:
        return

    ndim = labels_reshaped.ndim
    if ndim < 2:
        LOGGER.warning("无法绘制维度小于2的图像。")
        return

    if ndim == 2:
        slice_data = labels_reshaped
        title = 'KMeans Clustering Result (2D)'
        LOGGER.info("保存 2D 聚类结果 -> %s", output_path)
    else:  # 3D or higher
        # 规范化轴索引
        if axis < 0:
            axis += ndim
        axis = max(0, min(axis, ndim - 1))

        # 规范化切片索引
        max_index = labels_reshaped.shape[axis] - 1
        if slice_index is None or not (0 <= slice_index <= max_index):
            slice_index = labels_reshaped.shape[axis] // 2

        LOGGER.info("保存聚类图像切片 -> %s (axis=%s, slice=%s)", output_path, axis,
                    slice_index)
        
        # 使用slicing获取切片数据
        slicer = [slice(None)] * ndim
        slicer[axis] = slice_index
        slice_data = labels_reshaped[tuple(slicer)]
        title = f'KMeans Clustering Result (slice {slice_index} along axis {axis})'

    plt.figure(figsize=(8, 8))
    plt.imshow(slice_data, cmap='viridis', interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Cluster ID')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    LOGGER.info("聚类可视化结果已保存到 %s", output_path)


def configure_feature_extractor(
    image_is_2d: bool) -> featureextractor.RadiomicsFeatureExtractor:
    """
    根据影像维度，配置并返回一个PyRadiomics特征提取器。

    参数:
      image_is_2d: 一个布尔值，指示输入影像是否为2D。

    返回:
      一个配置好的 RadiomicsFeatureExtractor 实例。
    """
    LOGGER.info("初始化 PyRadiomics 特征提取器 (2D模式: %s)", image_is_2d)
    settings = {
        'binWidth': 25,
        'interpolator': 'sitkBSpline',
        'resampledPixelSpacing': None,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    # 启用在2D和3D下均有意义的特征类别
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')  # 灰度共生矩阵 (纹理特征)

    # 根据维度选择合适的形态学特征
    if image_is_2d:
        LOGGER.info("启用 2D 形态学特征 (shape2D)")
        extractor.enableFeatureClassByName('shape2D')
    else:
        LOGGER.info("启用 3D 形态学特征 (shape)")
        extractor.enableFeatureClassByName('shape')

    LOGGER.info("特征提取器配置完成")
    return extractor


def extract_cluster_features(
    image_obj: sitk.Image,
    labels_reshaped: np.ndarray,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    max_voxels_per_cluster: int,
    num_workers: int,
) -> Dict[int, Dict[str, Any]]:
    """
    对每个聚类区域并发执行PyRadiomics特征提取。

    参数:
      image_obj: 原始影像的SimpleITK对象。
      labels_reshaped: 经过后处理的、包含连续区域标签的NumPy数组。
      extractor: 配置好的PyRadiomics特征提取器实例。
      max_voxels_per_cluster: 单个区域允许的最大体素数。
      num_workers: 用于并发执行的线程数。

    返回:
      一个字典，键是聚类/区域的标签ID，值是包含该区域特征的字典。
    """
    cluster_features: Dict[int, Dict[str, Any]] = {}
    unique_labels = np.unique(labels_reshaped)
    # 排除背景标签0
    cluster_ids = sorted([label for label in unique_labels if label != 0])
    
    if not cluster_ids:
        LOGGER.warning("在掩码中未找到有效的非背景区域进行特征提取。")
        return {}

    def process_cluster(cluster_id: int):
        """为单个聚类提取特征的内部函数。"""
        cluster_mask_array = (labels_reshaped == cluster_id)
        voxel_count = int(np.sum(cluster_mask_array))

        if voxel_count > max_voxels_per_cluster:
            LOGGER.warning("  区域 %s 过大 (%s 体素)，跳过以避免内存问题。",
                           cluster_id, voxel_count)
            return None
        
        # 将NumPy掩码转换为带有正确元数据的SimpleITK图像
        cluster_mask_sitk = sitk.GetImageFromArray(cluster_mask_array.astype(np.uint8))
        cluster_mask_sitk.CopyInformation(image_obj)

        try:
            # PyRadiomics要求标签必须为整数类型
            feature_result = extractor.execute(image_obj, cluster_mask_sitk, label=1)
            # 移除PyRadiomics添加的诊断信息，只保留特征值
            cleaned_features = {
                k: v for k, v in feature_result.items() if not k.startswith('diagnostics')
            }
            return cluster_id, cleaned_features
        except Exception as exc:
            LOGGER.error("  区域 %s 特征提取失败: %s", cluster_id, exc, exc_info=True)
            return None

    LOGGER.info("开始对 %d 个区域进行特征提取...", len(cluster_ids))
    if num_workers > 1:
        LOGGER.info("启用多线程特征提取 (workers=%s)", num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {
                executor.submit(process_cluster, cluster_id): cluster_id
                for cluster_id in cluster_ids
            }
            for i, future in enumerate(as_completed(future_to_id)):
                cluster_id = future_to_id[future]
                LOGGER.info("  处理进度 %d/%d (区域ID: %s)", i + 1, len(cluster_ids), cluster_id)
                result = future.result()
                if result:
                    cluster_features[result[0]] = result[1]
    else:  # 单线程执行
        for i, cluster_id in enumerate(cluster_ids):
            LOGGER.info("  处理进度 %d/%d (区域ID: %s)", i + 1, len(cluster_ids), cluster_id)
            result = process_cluster(cluster_id)
            if result:
                cluster_features[result[0]] = result[1]

    LOGGER.info("特征提取完成，成功处理 %s 个区域。", len(cluster_features))
    return cluster_features


def summarize_features(cluster_features: Dict[int, Dict[str, Any]],
                       max_items_to_print: int = 5) -> None:
    """
    打印每个聚类提取出的特征摘要。

    参数:
      cluster_features: 包含所有区域特征的字典。
      max_items_to_print: 每个区域最多打印的特征数量。
    """
    LOGGER.info("--- 特征提取结果摘要 ---")
    for cluster_id, features in sorted(cluster_features.items()):
        LOGGER.info("区域 %s (共 %d 个特征):", cluster_id, len(features))
        for i, (name, value) in enumerate(features.items()):
            if i >= max_items_to_print:
                LOGGER.info("  ... 等 %d 个更多特征", len(features) - i)
                break
            # 对浮点数值进行格式化以提高可读性
            if isinstance(value, (float, np.floating)):
                LOGGER.info("  - %s: %.4f", name, value)
            else:
                LOGGER.info("  - %s: %s", name, value)
    LOGGER.info("--- 摘要结束 ---")


def run_pipeline(config: PipelineConfig) -> Dict[int, Dict[str, Any]]:
    """
    执行完整的影像分析流程。

    参数:
      config: 包含所有流程配置的PipelineConfig对象。

    返回:
      一个字典，包含每个成功处理的区域的影像组学特征。
    """
    # 1. 加载影像
    image_data, image_obj = load_medical_image(config.image_path,
                                               config.image_format)

    # 2. 执行K-Means聚类
    raw_labels = perform_kmeans_clustering(image_data, config.n_clusters,
                                           config.random_state)
    
    # 3. 后处理分割掩码以获得空间连续的区域
    processed_labels = _postprocess_segmentation_mask(raw_labels)

    # 4. 可视化聚类结果
    plot_cluster_slice(processed_labels, config.plot_path, config.slice_axis,
                       config.slice_index)

    # 5. 根据影像维度配置特征提取器
    is_2d = image_obj.GetDimension() == 2
    extractor = configure_feature_extractor(image_is_2d=is_2d)

    # 6. 提取特征
    cluster_features = extract_cluster_features(
        image_obj=image_obj,
        labels_reshaped=processed_labels,
        extractor=extractor,
        max_voxels_per_cluster=config.max_voxels_per_cluster,
        num_workers=config.num_workers,
    )

    # 7. 打印结果摘要
    if cluster_features:
        summarize_features(cluster_features)
    else:
        LOGGER.warning("流程执行完毕，但没有提取到任何特征。")

    return cluster_features


def parse_args() -> PipelineConfig:
    """解析命令行参数并返回一个PipelineConfig实例。"""
    parser = argparse.ArgumentParser(
        description="基于K-Means和PyRadiomics的医学影像特征提取流程。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image-path', default='./pyradiomics/data/37.png',
        help='待处理的医学影像路径。')
    parser.add_argument(
        '--image-format', default='auto',
        help='影像格式 (例如: sitk, auto)。')
    parser.add_argument(
        '--clusters', type=int, default=2,
        help='K-Means聚类的数量。')
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='K-Means的随机种子，用于复现性。')
    parser.add_argument(
        '--slice-axis', type=int, default=-1,
        help='3D影像可视化切片的轴 (0, 1, 2)。-1代表最后一个轴。')
    parser.add_argument(
        '--slice-index', type=int, default=None,
        help='可视化切片的索引。默认为中间层。')
    parser.add_argument(
        '--plot-path', default='clustering_result.png',
        help='聚类切片的可视化保存路径。输入 "none" 可禁用保存。')
    parser.add_argument(
        '--max-voxels', type=int, default=5_000_000,
        help='为防止内存耗尽，单个聚类区域允许处理的最大体素数量。')
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count() or 1,
        help='特征提取时使用的并发线程数。')
    args = parser.parse_args()

    # 特殊处理 "none" 字符串以禁用绘图
    plot_path = None if args.plot_path.lower() == 'none' else args.plot_path

    return PipelineConfig(
        image_path=args.image_path,
        image_format=args.image_format,
        n_clusters=args.clusters,
        random_state=args.random_state,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        plot_path=plot_path,
        max_voxels_per_cluster=args.max_voxels,
        num_workers=max(1, args.workers),
    )


def main():
    """程序主入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    try:
        config = parse_args()
        run_pipeline(config)
    except Exception as e:
        LOGGER.critical("流程执行过程中发生未捕获的异常: %s", e, exc_info=True)


if __name__ == '__main__':
    main()