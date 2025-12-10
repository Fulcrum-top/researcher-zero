"""
ResearcherZero 配置管理
只处理非模型的配置
"""
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

@dataclass
class MCPConfig:
    """MCP工具配置"""
    use_mcp: bool = True
    mcp_server_timeout: int = 30

@dataclass
class ArxivConfig:
    """arXiv配置"""
    max_results: int = 10
    sort_by: str = "relevance"
    sort_order: str = "descending"

@dataclass
class StorageConfig:
    """存储配置"""
    papers_dir: str = "data/papers"
    knowledge_dir: str = "data/knowledge"
    cache_dir: str = "data/cache"

    def __post_init__(self):
        # 确保目录存在
        for dir_path in [self.papers_dir, self.knowledge_dir, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)

            # 创建子目录
            for subdir in ["states", "papers", "reports", "errors"]:
                os.makedirs(f"{dir_path}/{subdir}", exist_ok=True)

@dataclass
class ToolConfig:
    """工具配置"""
    max_references: int = 50  # 最多处理的参考文献数量
    chunk_size: int = 2000    # 文本分块大小
    overlap_size: int = 200   # 分块重叠大小
    top_k_papers: int = 5     # 处理的Top K论文数量

@dataclass
class AgentConfig:
    """Agent配置"""
    default_model: str = "kimi-latest"
    temperature: float = 0.1
    max_tokens: int = 4000
    reasoning_depth: str = "deep"  # deep, medium, shallow

@dataclass
class Settings:
    """全局配置"""
    # 模块配置
    mcp: MCPConfig = MCPConfig()
    agent: AgentConfig = AgentConfig()
    arxiv: ArxivConfig = ArxivConfig()
    storage: StorageConfig = StorageConfig()
    tools: ToolConfig = ToolConfig()

    # 项目特定配置
    default_domain: str = "artificial intelligence"
    min_citation_count: int = 2  # 最低引用次数阈值

    @property
    def project_root(self) -> Path:
        """获取项目根目录"""
        return Path(__file__).parent.parent.parent

settings = Settings()