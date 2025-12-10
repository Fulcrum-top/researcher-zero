# conftest.py
import pytest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from core.gateways.llm_gateway import MultiModelClient
from core.tools.config_loader import ConfigLoader


@pytest.fixture
def sample_config_content():
    """返回示例配置内容 - 更新为新的模型名称"""
    return """
models:
  kimi-latest:
    provider: "openai"
    model_name: "kimi-latest"
    api_key: "test_kimi_key"
    base_url: "https://api.moonshot.cn/v1"
    timeout: 60
    max_retries: 3

  deepseek-chat:
    provider: "openai"
    model_name: "deepseek-chat"
    api_key: "test_deepseek_key"
    base_url: "https://api.deepseek.com/v1"
    timeout: 30

  qwen-turbo:
    provider: "openai"
    model_name: "qwen-turbo"
    api_key: "test_qwen_key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout: 30

  qwen-vl:
    provider: "openai"
    model_name: "qwen-vl"
    api_key: "test_qwen_key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout: 60

defaults:
  timeout: 30
  max_retries: 3
  retry_interval: 1
"""


@pytest.fixture
def sample_env_content():
    """返回示例环境变量内容"""
    return """KIMI_API_KEY=test_kimi_key
DEEPSEEK_API_KEY=test_deepseek_key
QWEN_API_KEY=test_qwen_key
LOG_LEVEL=INFO
"""


@pytest.fixture
def sample_config_file(sample_config_content):
    """创建临时的配置文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sample_config_content)
        temp_path = f.name

    yield temp_path
    # 测试完成后清理临时文件
    os.unlink(temp_path)


@pytest.fixture
def sample_env_file(sample_env_content):
    """创建临时的环境变量文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(sample_env_content)
        temp_path = f.name

    yield temp_path
    # 测试完成后清理临时文件
    os.unlink(temp_path)


@pytest.fixture
def mock_litellm_response():
    """创建模拟的LiteLLM响应"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "这是一个模拟回复"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    return mock_response


@pytest.fixture
def multi_model_client(sample_config_file, sample_env_file):
    """创建配置好的MultiModelClient实例"""
    with patch('litellm.completion') as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "测试回复"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_completion.return_value = mock_response

        client = MultiModelClient(sample_config_file, sample_env_file)
        yield client


@pytest.fixture
def config_loader(sample_config_file, sample_env_file):
    """创建配置好的ConfigLoader实例"""
    loader = ConfigLoader(config_path=sample_config_file, env_path=sample_env_file)
    return loader


@pytest.fixture(autouse=True)
def setup_test_environment():
    """自动为每个测试设置环境"""
    # 保存原始环境
    original_env = os.environ.copy()

    yield

    # 恢复原始环境
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_messages():
    """返回示例消息列表"""
    return [
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]


@pytest.fixture
def sample_image_messages():
    """返回包含图片的示例消息列表"""
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]


def pytest_configure(config):
    """Pytest配置钩子"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: 标记测试为慢速测试（会跳过除非明确指定）"
    )
    config.addinivalue_line(
        "markers", "integration: 标记测试为集成测试"
    )
    config.addinivalue_line(
        "markers", "unit: 标记测试为单元测试"
    )


def pytest_addoption(parser):
    """添加自定义命令行选项"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="运行慢速测试"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="运行集成测试"
    )


def pytest_collection_modifyitems(config, items):
    """根据命令行选项修改测试项"""
    # 如果没有指定--run-slow，跳过标记为slow的测试
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="需要 --run-slow 选项来运行")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # 如果没有指定--run-integration，跳过标记为integration的测试
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="需要 --run-integration 选项来运行")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)