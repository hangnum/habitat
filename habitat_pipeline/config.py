"""配置定义模块。"""

from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    """生境特征提取配置。

    Attributes:
        n_clusters: K-Means 聚类的簇数。
        bin_width: 灰度直方图的固定 bin 宽度。
        min_roi_pixels: ROI 最少像素数量，低于该值时放弃该图像。
    """

    n_clusters: int = 3
    bin_width: float = 25.0
    min_roi_pixels: int = 10

