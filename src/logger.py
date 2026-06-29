import logging
import sys
from pathlib import Path

def setup_logger(name:str = "cs_ai", log_file:str = "app.log"):
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # 控制台输出（带颜色）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '\033[92m%(asctime)s\033[0m - \033[94m%(name)s\033[0m - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    console.setFormatter(console_format)
    logger.addHandler(console)
    # 文件输出（详细）
    file_handler = logging.FileHandler(log_dir / log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()


