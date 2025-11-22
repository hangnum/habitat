"""日志与第三方静音配置。"""

import logging

from pyradiomics.radiomics import featureextractor


def setup_logging(level: int = logging.INFO, quiet: bool = False) -> logging.Logger:
    """初始化日志，并屏蔽 PyRadiomics 的冗余输出。

    Args:
        level: 主流程的日志等级。
        quiet: 当为 True 时，仅输出 ERROR 及以上级别。

    Returns:
        logging.Logger: 已配置好的顶级 logger。
    """
    effective_level = logging.ERROR if quiet else level
    logging.basicConfig(
        level=effective_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("habitat")

    for name in ("radiomics", "radiomics.featureextractor"):
        rlog = logging.getLogger(name)
        rlog.handlers.clear()
        rlog.setLevel(logging.CRITICAL)
        rlog.propagate = False

    # 双保险：禁用模块内置 logger，避免后续重新添加 handler。
    featureextractor.logger.disabled = True

    return logger

