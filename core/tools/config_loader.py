# config_loader.py
import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

# 使用项目内的日志系统
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logs import logger


def get_project_root():
    """获取项目根目录（Fulcrum_ai文件夹）"""
    # 当前文件路径: Fulcrum_ai/src/tools/config_loader.py
    current_file = os.path.abspath(__file__)
    # 项目根目录: Fulcrum_ai/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    return project_root


class ConfigLoader:
    """配置加载器，支持环境变量替换"""

    def __init__(self, config_path: str = None, env_path: str = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
            env_path: 环境变量文件路径，如果为None则使用默认路径
        """
        # 设置默认路径
        if config_path is None:
            project_root = get_project_root()
            config_path = os.path.join(project_root, "src", "configs", "models.yaml")

        if env_path is None:
            project_root = get_project_root()
            env_path = os.path.join(project_root, ".env")

        self.config_path = config_path
        self.env_path = env_path

        logger.info(f"配置加载器初始化 - 配置文件: {self.config_path}")
        logger.info(f"配置加载器初始化 - 环境文件: {self.env_path}")

        self._load_environment_variables()
        self.config = self._load_config()

    def _load_environment_variables(self):
        """加载环境变量文件到系统环境变量"""
        if os.path.exists(self.env_path):
            load_dotenv(self.env_path)
            logger.info(f"已加载环境变量文件: {self.env_path}")
        else:
            logger.warning(f"环境变量文件未找到: {self.env_path}，将使用系统环境变量")

    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件并解析环境变量"""
        logger.info(f"尝试加载配置文件: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_content = file.read()
                # 替换环境变量
                config_content = self._replace_env_vars(config_content)
                config_data = yaml.safe_load(config_content)
                logger.info(f"配置文件加载成功，找到 {len(config_data.get('models', {}))} 个模型")
                return config_data
        except FileNotFoundError:
            logger.error(f"配置文件未找到: {self.config_path}")
            # 返回默认配置结构
            return {
                'models': {},
                'defaults': {
                    'timeout': 30,
                    'max_retries': 3,
                    'retry_interval': 1
                }
            }
        except yaml.YAMLError as e:
            logger.error(f"YAML解析错误: {e}")
            raise

    def _replace_env_vars(self, content: str) -> str:
        """替换 ${VAR} 格式的环境变量"""
        import re
        pattern = r'\$\{([^}]+)\}'

        def replace_match(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                logger.warning(f"环境变量 {var_name} 未设置，将使用空值")
                return ""
            return value

        return re.sub(pattern, replace_match, content)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """获取特定模型的配置"""
        model_config = self.config['models'].get(model_name)
        if not model_config:
            raise ValueError(f"模型配置未找到: {model_name}")

        # 合并默认配置
        defaults = self.config.get('defaults', {})
        return {**defaults, **model_config}

    def list_available_models(self) -> list:
        """获取所有可用的模型列表"""
        return list(self.config['models'].keys())

    def validate_config(self):
        """验证配置是否正确加载"""
        missing_vars = []
        for model_name, config in self.config['models'].items():
            api_key = config.get('api_key', '')
            if isinstance(api_key, str) and api_key.startswith('${') and api_key.endswith('}'):
                var_name = api_key[2:-1]
                if not os.getenv(var_name):
                    missing_vars.append(f"{model_name}: {var_name}")

        if missing_vars:
            raise ValueError(f"以下环境变量未设置: {', '.join(missing_vars)}")

        logger.info("配置验证通过")