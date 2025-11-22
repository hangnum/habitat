# 医学生境分析管道

基于 PyRadiomics 的肝癌生境特征提取工具，提供命令行入口、可配置的 K-Means 生境划分及 CSV 导出。

## 目录结构

- `main.py`：CLI 入口，转发到 `habitat_pipeline` 包。
- `habitat_pipeline/`：核心代码，包含配置、特征提取、数据组织与管线调度。
- `pyradiomics/`：精简后的第三方依赖，仅保留运行所需的 `radiomics` 源码与元数据。
- `.gitignore`：忽略缓存与医学影像常见大文件。

## 使用

1. 安装依赖（确保本地环境已安装 SimpleITK、scikit-learn、pandas、numpy 等）。
2. 运行命令：

```bash
python main.py \
  --data-dir /home/wwt/data/raw/liver/Grade \
  --output-dir /home/wwt/data/outputs/habitat \
  --clusters 3 \
  --bin-width 25 \
  --min-roi-pixels 10
```

常用参数：

- `--data-dir`：输入数据根目录，期望结构为 `gradeX/<patient>/<modality>/*.png`。
- `--output-dir`：结果 CSV 输出目录。
- `--clusters`：生境 K-Means 簇数。
- `--bin-width`：PyRadiomics 灰度 bin 宽度。
- `--min-roi-pixels`：ROI 最小像素数阈值，低于阈值会跳过。
- `--quiet`：静默模式，仅输出错误日志。

## 设计要点

- **模块化**：配置、日志、数据组织、特征提取与调度分层，便于扩展与单测。
- **稳健性**：影像统一强制 `(1.0, 1.0)` 间距；ROI 像素阈值防止聚类异常；PyRadiomics 日志静音避免刷屏。
- **依赖精简**：移除 PyRadiomics 的示例、测试、notebook 与实验脚本，仅保留运行所需源码与元数据。

