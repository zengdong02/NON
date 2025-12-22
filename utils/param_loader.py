import yaml
import os
from pathlib import Path


def load_config() -> dict:
    """
    加载并解析 YAML 配置文件。

    Args:
        config_path (str): 配置文件的路径。

    Returns:
        dict: 包含配置内容的字典。
    
    Raises:
        FileNotFoundError: 如果文件不存在。
        yaml.YAMLError: 如果文件格式解析错误。
    """
    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parent.parent
    config_path = PROJECT_ROOT / "configs/config.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"错误：配置文件 {config_path} 未找到！")
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML 解析错误: {e}")
        
    return config
