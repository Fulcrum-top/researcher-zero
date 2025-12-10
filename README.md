# ResearcherZero

ResearcherZero 是一个学术研究助手，旨在解决研究人员面临的"信息过载但认知不足"的矛盾。它通过持续学习构建领域认知图谱，让AI代理像研究员一样学习和理解领域知识。

## 核心功能

- **持续学习**：一次性深度扫荡 + 周期性增量学习
- **知识结晶化**：从论文中提取结构化知识，构建认知框架
- **状态保持**：每次交互都站在历史分析的基础上

## 技术架构

- 基于LangGraph的状态机实现单一Agent
- 使用FASTAPI构建应用
- 通过MCP协议暴露工具能力
- 多模型支持：Kimi、DeepSeek、通义千问等

## 安装和配置

### 环境要求

- Python 3.8+
- pip包管理器

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

1. 复制示例环境变量文件：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，设置以下必要配置：
   - 各模型的API密钥
   - 外部LLM网关地址 (`OPENAI_BASE_URL`)
   
   示例配置：
   ```
   OPENAI_BASE_URL=http://localhost:8000
   KIMI_API_KEY=sk-xxx
   DEEPSEEK_API_KEY=sk-xxx
   ```

### 启动外部LLM网关

请参考 [llm-gateway](https://github.com/Fulcrum-top/llm-gateway) 仓库启动外部网关服务：

```bash
git clone https://github.com/Fulcrum-top/llm-gateway
cd llm-gateway
cp .env.example .env
# 编辑.env文件设置API密钥
uv run litellm --config config.yaml
```

## 使用方法

```bash
# 学习指定领域
python main.py --domain "reinforcement learning"

# 指定模型
python main.py --domain "computer vision" --model deepseek-chat

# 查看可用模型
python main.py --list-models
```

## 配置

通过环境变量 `VERBOSE` 控制日志启用/禁用：
- `VERBOSE=true` 或 `VERBOSE=1`：启用（默认）

## 项目结构

```
researcher-zero/
├── core/
│   ├── agents/           # Agent实现
│   ├── configs/          # 配置文件
│   ├── infra/            # 基础设施
│   ├── mcp/              # MCP协议实现
│   ├── tools/            # 工具实现
│   └── utils/            # 工具类
├── data/                 # 数据目录
│   ├── cache/            # 缓存
│   └── knowledge/        # 知识输出
├── logs/                 # 日志目录
├── tests/                # 测试代码
├── main.py               # 主程序入口
├── requirements.txt      # 依赖列表
└── .env.example          # 环境变量示例
```