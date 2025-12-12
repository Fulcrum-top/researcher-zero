import operator
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import MessageLikeRepresentation
from typing import Annotated


def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


# ==================== 数据模型定义 ====================

class PaperMetadata(BaseModel):
    """论文元数据结构"""
    title: str = Field(description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    abstract: str = Field(default="", description="摘要")
    year: int = Field(default=0, description="发表年份")
    venue: str = Field(default="", description="会议/期刊")
    citation_count: Optional[int] = Field(default=None, description="引用数")
    doi: Optional[str] = Field(default=None, description="DOI")
    url: str = Field(default="", description="论文链接")
    pdf_url: Optional[str] = Field(default=None, description="PDF链接")
    source: str = Field(default="", description="数据源")
    is_survey: bool = Field(default=False, description="是否为综述论文")
    citations: List[str] = Field(default_factory=list, description="引用文献列表")
    relevance_score: float = Field(default=0.0, description="相关性分数")
    source_id: str = Field(default="", description="原始数据源ID")


class SearchPlan(BaseModel):
    """搜索计划"""
    primary_goal: str = Field(default="survey", description="主要目标：survey 或 empirical")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    time_filter: Optional[str] = Field(default=None, description="时间过滤")
    must_include: List[str] = Field(default_factory=list, description="必须包含的词")
    exclude: List[str] = Field(default_factory=list, description="排除的词")
    expected_count: int = Field(default=20, description="期望返回数量")


class SearchStrategy(BaseModel):
    """搜索执行策略"""
    sources: List[str] = Field(default_factory=list, description="使用的数据源")
    query_params: Dict[str, Any] = Field(default_factory=dict, description="各数据源查询参数")
    max_results_per_source: int = Field(default=20, description="每个源的最大结果数")
    prioritize_surveys: bool = Field(default=True, description="是否优先综述论文")
    iteration: int = Field(default=0, description="当前迭代次数")


class QualityMetrics(BaseModel):
    """质量评估指标"""
    breadth_score: float = Field(default=0.0, description="广度分数")
    depth_score: float = Field(default=0.0, description="深度分数")
    timeliness_score: float = Field(default=0.0, description="时间有效性分数")
    overall_score: float = Field(default=0.0, description="总体质量分")
    issues: List[str] = Field(default_factory=list, description="存在的问题")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


# ==================== 工具调用定义 ====================

class ConductArxivSearch(BaseModel):
    """调用arXiv搜索工具"""
    query: str = Field(description="搜索查询")
    max_results: int = Field(default=20, description="最大结果数")
    category: Optional[str] = Field(default=None, description="arXiv分类")


class ConductSemanticScholarSearch(BaseModel):
    """调用Semantic Scholar搜索工具"""
    query: str = Field(description="搜索查询")
    max_results: int = Field(default=20, description="最大结果数")
    year_filter: Optional[str] = Field(default=None, description="年份过滤")


class GetPaperMetadata(BaseModel):
    """获取论文详细元数据"""
    paper_id: str = Field(description="论文ID")
    source: str = Field(default="semantic_scholar", description="数据源")


class GetCitationList(BaseModel):
    """获取引用文献列表"""
    paper_id: str = Field(description="论文ID")
    max_citations: int = Field(default=50, description="最大引用数")
    source: str = Field(default="semantic_scholar", description="数据源")


class DownloadPaperPDF(BaseModel):
    """下载论文PDF"""
    paper_id: str = Field(description="论文ID")
    source: str = Field(default="arxiv", description="数据源")


class SearchComplete(BaseModel):
    """完成搜索"""


# ==================== 主要状态类 ====================

class PaperSearcherState(MessagesState):
    """论文搜索Agent的状态"""

    # 输入
    user_query: str = ""
    raw_query: str = ""

    # 分析层
    search_plan: Optional[SearchPlan] = None
    search_strategy: Optional[SearchStrategy] = None

    # 执行层
    search_iteration: int = 0
    quality_threshold: float = Field(default=0.7, description="质量阈值")

    # 工具调用结果
    raw_search_results: Dict[str, List[Dict]] = Field(default_factory=dict, description="原始搜索结果")
    merged_papers: List[PaperMetadata] = Field(default_factory=list, description="合并去重后的论文")
    papers_with_citations: List[PaperMetadata] = Field(default_factory=list, description="带引用列表的论文")

    # 处理层
    ranked_papers: List[PaperMetadata] = Field(default_factory=list, description="排序后的论文")
    quality_metrics: Optional[QualityMetrics] = None

    # 历史记录
    search_history: List[Dict[str, Any]] = Field(default_factory=list, description="搜索历史")

    # 输出层
    final_results: List[PaperMetadata] = Field(default_factory=list, description="最终结果")
    final_report: Optional[str] = None
    reasoning: Optional[str] = None

    # Agent消息
    searcher_messages: Annotated[list[MessageLikeRepresentation], override_reducer] = Field(default_factory=list)