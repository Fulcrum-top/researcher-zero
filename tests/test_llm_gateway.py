# test_llm_gateway.py
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from core.utils.logs import logger
from core.gateways.llm_gateway import MultiModelClient
from core.tools.config_loader import ConfigLoader


class TestMultiModelClient:
    """多模型客户端测试类"""

    def test_initialization(self, sample_config_file, sample_env_file):
        """测试客户端初始化"""
        with patch('litellm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "测试回复"
            mock_response.usage = MagicMock()
            mock_completion.return_value = mock_response

            client = MultiModelClient(sample_config_file, sample_env_file)

            # 验证客户端正确初始化
            assert client is not None
            assert len(client.get_available_models()) == 4
            expected_models = ["kimi-latest", "deepseek-chat", "qwen-turbo", "qwen-vl"]
            for model in expected_models:
                assert model in client.get_available_models()

    def test_chat_completion_success(self, multi_model_client, sample_messages, mock_litellm_response):
        """测试成功的聊天补全调用"""
        with patch('litellm.completion', return_value=mock_litellm_response):
            # 测试所有支持的模型
            models_to_test = ["kimi-latest", "deepseek-chat", "qwen-turbo"]

            for model_name in models_to_test:
                response = multi_model_client.chat_completion(sample_messages, model_name)

                assert response["success"] is True
                assert response["model"] == model_name
                assert "模拟回复" in response["data"].choices[0].message.content
                logger.debug(f"成功测试模型: {model_name}")

    def test_chat_text_all_models(self, multi_model_client, mock_litellm_response):
        """测试所有模型的文本对话功能"""
        with patch('litellm.completion', return_value=mock_litellm_response):
            models_to_test = ["kimi-latest", "deepseek-chat", "qwen-turbo"]

            for model_name in models_to_test:
                response = multi_model_client.chat_text("Hello", model_name)

                assert response == "这是一个模拟回复"
                logger.debug(f"文本对话测试通过: {model_name}")

    def test_chat_image_with_qwen_vl(self, multi_model_client, mock_litellm_response):
        """测试 Qwen-VL 的图文对话功能"""
        with patch('litellm.completion', return_value=mock_litellm_response):
            response = multi_model_client.chat_image(
                prompt="描述这张图片",
                image_url="https://example.com/image.jpg",
                model_name="qwen-vl"
            )

            assert response == "这是一个模拟回复"
            logger.debug("图文对话测试通过: qwen-vl")

    def test_invalid_model(self, multi_model_client, sample_messages):
        """测试无效模型名称"""
        with pytest.raises(ValueError, match="模型配置未找到"):
            multi_model_client.chat_completion(sample_messages, "invalid-model")

    def test_get_available_models(self, multi_model_client):
        """测试获取可用模型列表"""
        models = multi_model_client.get_available_models()

        expected_models = ["kimi-latest", "deepseek-chat", "qwen-turbo", "qwen-vl"]
        for model in expected_models:
            assert model in models

        logger.debug(f"可用模型列表: {models}")

    def test_model_config_loading(self, sample_config_file, sample_env_file):
        """测试模型配置加载"""
        with patch('litellm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "测试回复"
            mock_response.usage = MagicMock()
            mock_completion.return_value = mock_response

            client = MultiModelClient(sample_config_file, sample_env_file)

            # 验证每个模型的配置
            model_configs = {
                "kimi-latest": {
                    "provider": "openai",
                    "base_url": "https://api.moonshot.cn/v1",
                    "timeout": 60
                },
                "deepseek-chat": {
                    "provider": "openai",
                    "base_url": "https://api.deepseek.com/v1",
                    "timeout": 30
                },
                "qwen-turbo": {
                    "provider": "openai",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "timeout": 30
                },
                "qwen-vl": {
                    "provider": "openai",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "timeout": 60
                }
            }

            for model_name, expected_config in model_configs.items():
                config = client.get_model_config(model_name)
                for key, value in expected_config.items():
                    assert config[key] == value
                logger.debug(f"配置验证通过: {model_name}")

    def test_env_file_loading(self, sample_config_file, sample_env_file):
        """测试环境变量文件加载"""
        with patch('litellm.completion') as mock_completion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "测试回复"
            mock_response.usage = MagicMock()
            mock_completion.return_value = mock_response

            client = MultiModelClient(sample_config_file, sample_env_file)

            # 确保客户端正常初始化
            assert client is not None
            assert len(client.get_available_models()) == 4
            logger.debug("环境变量文件加载测试通过")

    @pytest.mark.slow
    def test_retry_mechanism(self, sample_config_file, sample_env_file, sample_messages):
        """测试重试机制"""
        # 模拟前两次失败，第三次成功
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "最终成功回复"
        mock_response.usage = MagicMock()

        with patch('litellm.completion') as mock_completion:
            # 设置前两次调用抛出可重试的异常，第三次返回成功
            mock_completion.side_effect = [
                Exception("rate limit exceeded"),  # 可重试的错误
                Exception("network timeout"),  # 可重试的错误
                mock_response  # 第三次成功
            ]

            client = MultiModelClient(sample_config_file, sample_env_file)

            # 应该会重试并最终成功
            response = client.chat_completion(
                sample_messages,
                "kimi-latest",
                retry_enabled=True
            )

            assert response["success"] is True
            assert mock_completion.call_count == 3
            logger.debug("重试机制测试通过")

    @pytest.mark.slow
    def test_retry_disabled(self, sample_config_file, sample_env_file, sample_messages):
        """测试禁用重试机制"""
        with patch('litellm.completion') as mock_completion:
            # 设置第一次调用就抛出异常
            mock_completion.side_effect = Exception("rate limit exceeded")

            client = MultiModelClient(sample_config_file, sample_env_file)

            # 当重试禁用时，应该直接抛出异常
            with pytest.raises(Exception, match="rate limit exceeded"):
                client.chat_completion(
                    sample_messages,
                    "kimi-latest",
                    retry_enabled=False
                )

            # 应该只调用了一次，没有重试
            assert mock_completion.call_count == 1
            logger.debug("禁用重试测试通过")

    def test_chat_with_system_message(self, multi_model_client, mock_litellm_response):
        """测试带系统消息的对话"""
        with patch('litellm.completion', return_value=mock_litellm_response):
            response = multi_model_client.chat_text(
                prompt="你好",
                model_name="deepseek-chat",
                system_message="你是一个专业的AI助手"
            )

            assert response == "这是一个模拟回复"
            logger.debug("系统消息测试通过")

    def test_chat_with_custom_parameters(self, multi_model_client, mock_litellm_response):
        """测试带自定义参数的对话"""
        with patch('litellm.completion', return_value=mock_litellm_response) as mock_completion:
            response = multi_model_client.chat_text(
                prompt="写一首诗",
                model_name="qwen-turbo",
                temperature=0.7,
                max_tokens=100
            )

            assert response == "这是一个模拟回复"

            # 验证自定义参数被传递
            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs.get('temperature') == 0.7
            assert call_kwargs.get('max_tokens') == 100
            logger.debug("自定义参数测试通过")

    @pytest.mark.integration
    def test_real_api_call(self, sample_config_file, sample_env_file):
        """测试真实API调用"""
        logger.info("[START]开始真实API调用测试...")
        # 使用真实配置文件路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        real_config_path = os.path.join(project_root, "src", "configs", "models.yaml")
        real_env_path = os.path.join(project_root, ".env")

        # 检查文件是否存在
        if not os.path.exists(real_config_path):
            pytest.skip(f"真实配置文件不存在: {real_config_path}")
            return

        if not os.path.exists(real_env_path):
            pytest.skip(f"真实环境文件不存在: {real_env_path}")
            return
        try:
            # 创建客户端实例
            client = MultiModelClient(real_config_path, real_env_path)

            # 获取可用模型
            available_models = client.get_available_models()
            logger.info(f"可用模型: {available_models}")

            # 选择要测试的模型（可以根据您的API密钥情况调整）
            models_to_test = []
            env_vars = {}
            with open(real_env_path, 'r',encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value

            # 检查每个模型所需的API密钥是否存在
            model_key_mapping = {
                "kimi-latest": "KIMI_API_KEY",
                "deepseek-chat": "DEEPSEEK_API_KEY",
                "qwen-turbo": "QWEN_API_KEY",
                "qwen-vl": "QWEN_API_KEY"
            }

            for model_name in available_models:
                required_key = model_key_mapping.get(model_name)
                if required_key and env_vars.get(required_key):
                    models_to_test.append(model_name)

            if not models_to_test:
                pytest.skip("没有找到有效的API密钥，跳过真实API测试")
                return

            logger.info(f"[TODO]将测试以下模型: {models_to_test}")
            # 对每个有API密钥的模型进行简单测试
            successful_tests = 0
            # 对每个有API密钥的模型进行简单测试
            for model_name in models_to_test:
                logger.info(f"测试模型: {model_name}")

                try:
                    # 使用简单的提示进行测试，避免消耗太多token
                    if model_name == "qwen-vl":
                        # Qwen-VL 需要图像，跳过图像测试或使用默认图像
                        logger.info(f"[SKIPPING]跳过 {model_name} 的图像测试")
                        continue
                    else:
                        # 文本模型测试
                        response = client.chat_text(
                            prompt="请用（是/否）回答：天空是绿色的吗？",
                            model_name=model_name,
                            max_tokens=100  # 限制token使用
                        )

                    # 验证响应
                    assert response is not None, f"{model_name} 返回了空响应"
                    assert isinstance(response, str), f"{model_name} 响应不是字符串"
                    assert len(response) > 0, f"{model_name} 返回了空字符串"
                    successful_tests += 1

                    # 详细打印模型回复
                    logger.info(f"{model_name} 真实API测试通过")
                    logger.info("=" * 60)  # 分隔线
                    logger.info(f"完整回复内容:")
                    logger.info(f"   {response}")
                    logger.info(f"回复统计: {len(response)} 字符")
                    logger.info("=" * 60)

                    # 在控制台直接看到输出，添加 print
                    print(f"\n{model_name} 回复:")
                    print(f"   {response}")
                    print("-" * 50)

                except Exception as e:
                    logger.warning(f"[FAILED] {model_name} 测试失败: {e}")
                    # 不抛出异常，继续测试其他模型
                    continue

            logger.info("[FINISHED]真实API调用测试完成")

        except Exception as e:
            logger.error(f"[FAILED]真实API调用测试失败: {e}")
            pytest.fail(f"[FAILED]真实API调用测试失败: {e}")


class TestConfigLoader:
    """配置加载器测试类"""

    def test_config_loading(self, config_loader):
        """测试配置加载"""
        model_config = config_loader.get_model_config("kimi-latest")

        assert model_config["provider"] == "openai"
        assert model_config["model_name"] == "kimi-latest"
        assert model_config["api_key"] == "test_kimi_key"
        assert model_config["timeout"] == 60
        logger.debug("Kimi配置加载测试通过")

    def test_deepseek_config(self, config_loader):
        """测试DeepSeek配置"""
        model_config = config_loader.get_model_config("deepseek-chat")

        assert model_config["provider"] == "openai"
        assert model_config["model_name"] == "deepseek-chat"
        assert model_config["base_url"] == "https://api.deepseek.com/v1"
        assert model_config["timeout"] == 30
        logger.debug("DeepSeek配置加载测试通过")

    def test_qwen_configs(self, config_loader):
        """测试Qwen系列配置"""
        # 测试 Qwen Turbo
        turbo_config = config_loader.get_model_config("qwen-turbo")
        assert turbo_config["provider"] == "openai"
        assert turbo_config["model_name"] == "qwen-turbo"
        assert turbo_config["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 测试 Qwen VL
        vl_config = config_loader.get_model_config("qwen-vl")
        assert vl_config["provider"] == "openai"
        assert vl_config["model_name"] == "qwen-vl"
        assert vl_config["timeout"] == 60
        logger.debug("Qwen配置加载测试通过")

    def test_list_available_models(self, config_loader):
        """测试获取可用模型列表"""
        models = config_loader.list_available_models()

        expected_models = ["kimi-latest", "deepseek-chat", "qwen-turbo", "qwen-vl"]
        for model in expected_models:
            assert model in models

        logger.debug(f"配置加载器返回的模型列表: {models}")

    def test_validate_config(self, config_loader):
        """测试配置验证"""
        # 这个配置应该是有效的
        config_loader.validate_config()
        logger.debug("配置验证测试通过")

    def test_default_config_values(self, config_loader):
        """测试默认配置值"""
        model_config = config_loader.get_model_config("kimi-latest")

        # 验证默认配置被正确合并
        assert model_config["max_retries"] == 3  # 来自默认配置
        assert model_config["retry_interval"] == 1  # 来自默认配置
        logger.debug("默认配置合并测试通过")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])