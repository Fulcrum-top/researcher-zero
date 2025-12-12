import os
from typing import Any
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class PaperSearcherConfig(BaseModel):
    """论文搜索Agent的配置"""

    # Pipeline parameters
    max_search_iterations: int = Field(
        default=3,
        description="最大搜索迭代次数"
    )
    max_search_results: int = Field(
        default=20,
        description="最大搜索结果数"
    )
    quality_threshold: float = Field(
        default=0.7,
        description="结果质量阈值"
    )
    enable_pdf_download: bool = Field(
        default=True,
        description="是否启用PDF下载"
    )

    # 数据源配置
    enable_arxiv: bool = Field(
        default=True,
        description="是否启用arXiv搜索"
    )
    enable_semantic_scholar: bool = Field(
        default=True,
        description="是否启用Semantic Scholar搜索"
    )

    # Model parameters
    query_analysis_model: str = Field(
        default="kimi",
        description="查询分析使用的模型"
    )
    strategy_planning_model: str = Field(
        default="kimi",
        description="策略规划使用的模型"
    )
    quality_assessment_model: str = Field(
        default="kimi",
        description="质量评估使用的模型"
    )
    report_generation_model: str = Field(
        default="kimi",
        description="报告生成使用的模型"
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "PaperSearcherConfig":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})