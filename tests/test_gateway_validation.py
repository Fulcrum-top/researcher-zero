# test_gateway_validation.py
#!/usr/bin/env python3
"""
快速验证修复后的代码
"""

import sys
import os
import tempfile

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.logs import logger
from core.gateways.llm_gateway import MultiModelClient
from core.tools.config_loader import ConfigLoader
from unittest.mock import patch, MagicMock


def get_project_root():
    """获取项目根目录（Fulcrum_ai文件夹）"""
    # 当前文件路径: Fulcrum_ai/tests/test_gateway_validation.py
    current_file = os.path.abspath(__file__)
    # 项目根目录: Fulcrum_ai/
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root


def create_sample_config():
    """创建示例配置文件内容"""
    return """
models:
  kimi-latest:
    provider: openai
    model_name: kimi-latest
    api_key: test_kimi_key
    base_url: https://api.moonshot.cn/v1
    timeout: 60
    max_retries: 3
  deepseek-chat:
    provider: openai
    model_name: deepseek-chat
    api_key: test_deepseek_key
    base_url: https://api.deepseek.com/v1
    timeout: 30
  qwen-turbo:
    provider: openai
    model_name: qwen-turbo
    api_key: test_qwen_key
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    timeout: 30
  qwen-vl:
    provider: openai
    model_name: qwen-vl
    api_key: test_qwen_key
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    timeout: 60
defaults:
  timeout: 30
  max_retries: 3
  retry_interval: 1
"""


def create_sample_env():
    """创建示例环境变量文件内容"""
    return """KIMI_API_KEY=test_kimi_key
DEEPSEEK_API_KEY=test_deepseek_key
QWEN_API_KEY=test_qwen_key
"""


def test_with_actual_config():
    """使用实际配置文件进行测试"""
    logger.info("[START]使用实际配置文件测试...")

    try:
        # 获取实际文件路径
        project_root = get_project_root()
        actual_config_path = os.path.join(project_root, "src", "configs", "models.yaml")
        actual_env_path = os.path.join(project_root, ".env")

        logger.info(f"项目根目录: {project_root}")
        logger.info(f"实际配置文件路径: {actual_config_path}")
        logger.info(f"实际环境文件路径: {actual_env_path}")

        # 检查文件是否存在
        if not os.path.exists(actual_config_path):
            logger.error(f"配置文件不存在: {actual_config_path}")
            logger.info("请确保配置文件位于: src/configs/models.yaml")
            return False

        if not os.path.exists(actual_env_path):
            logger.warning(f"环境变量文件不存在: {actual_env_path}")
            logger.info("将使用系统环境变量")

        # 测试 ConfigLoader
        logger.info("1. 测试 ConfigLoader...")
        loader = ConfigLoader(config_path=actual_config_path, env_path=actual_env_path)
        models = loader.list_available_models()
        logger.info(f"[SUCCESS]ConfigLoader 初始化成功，找到 {len(models)} 个模型: {models}")

        # 测试 MultiModelClient
        logger.info("2. 测试 MultiModelClient...")
        with patch('litellm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "测试回复"
            mock_response.usage = MagicMock()
            mock_completion.return_value = mock_response

            client = MultiModelClient(config_path=actual_config_path, env_path=actual_env_path)
            models = client.get_available_models()
            logger.info(f"[SUCCESS]MultiModelClient 初始化成功，找到 {len(models)} 个模型: {models}")

            # 添加断言
            assert len(models) > 0, "应该至少找到一个模型"

            # 测试方法调用
            response = client.chat_text("测试", "kimi-latest")
            logger.info(f"[SUCCESS]chat_text 方法调用成功: {response}")
            assert  response is not None, "响应不应该为空"
        logger.info("[PASS]实际配置文件测试通过！")
        # return True

    except Exception as e:
        logger.error(f"[FAILED]实际配置文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_temp_files():
    """使用临时文件进行测试"""
    logger.info("[START]使用临时文件测试...")

    try:
        # 创建临时配置文件和环境变量文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            config_file.write(create_sample_config())
            config_path = config_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as env_file:
            env_file.write(create_sample_env())
            env_path = env_file.name

        try:
            # 测试 ConfigLoader
            logger.info("1. 测试 ConfigLoader...")
            loader = ConfigLoader(config_path=config_path, env_path=env_path)
            models = loader.list_available_models()
            logger.info(f"[SUCCESS]ConfigLoader 初始化成功，找到 {len(models)} 个模型: {models}")

            # 测试 MultiModelClient
            logger.info("2. 测试 MultiModelClient...")
            with patch('litellm.completion') as mock_completion:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "测试回复"
                mock_response.usage = MagicMock()
                mock_completion.return_value = mock_response

                client = MultiModelClient(config_path=config_path, env_path=env_path)
                models = client.get_available_models()
                logger.info(f"[SUCCESS]MultiModelClient 初始化成功，找到 {len(models)} 个模型: {models}")

                assert len(models) > 0, "应该至少找到一个模型"
                # 测试方法调用
                response = client.chat_text("测试", "kimi-latest")
                logger.info(f"[SUCCESS]chat_text 方法调用成功: {response}")

                assert  response is not None, "响应不应该为空"
            logger.info("[PASS]临时文件测试通过！")
            # return True

        finally:
            # 清理临时文件
            os.unlink(config_path)
            os.unlink(env_path)

    except Exception as e:
        logger.error(f"[FAILED]临时文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_default_paths():
    """测试默认路径"""
    logger.info("[START]测试默认路径...")

    try:
        # 测试使用默认路径
        logger.info("1. 测试 ConfigLoader 默认路径...")
        loader = ConfigLoader()
        logger.info(f"ConfigLoader 默认配置文件路径: {loader.config_path}")
        logger.info(f"ConfigLoader 默认环境文件路径: {loader.env_path}")

        logger.info("2. 测试 MultiModelClient 默认路径...")
        client = MultiModelClient()
        logger.info(f"MultiModelClient 默认配置文件路径: {client.config_path}")
        logger.info(f"MultiModelClient 默认环境文件路径: {client.env_path}")

        assert client.config_path is not None, "配置文件路径不应该为空"
        logger.info("[PASS]默认路径测试完成！")
        # return True

    except Exception as e:
        logger.error(f"[FAILED]默认路径测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 测试默认路径
    success0 = test_default_paths()
    print()

    # 测试实际配置文件
    success1 = test_with_actual_config()
    print()

    # 测试临时文件
    success2 = test_with_temp_files()

    if success0 and success1 and success2:
        logger.info("[PASS]所有测试通过！")
        sys.exit(0)
    else:
        logger.error("[FAILED]部分测试失败")
        sys.exit(1)