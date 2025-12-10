"""
学习模块的状态定义
核心理念：状态是Agent工作的上下文和记忆的载体
"""
from typing import TypedDict, List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class PaperMetadata:
    """论文元数据结构"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published: str
    categories: List[str]
    citation_count: int = 0
    reference_count: int = 0
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "arxiv_id": self.arxiv_id,
            "published": self.published,
            "categories": self.categories,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "relevance_score": self.relevance_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperMetadata':
        """从字典创建"""
        return cls(**data)


@dataclass
class ProcessedPaper:
    """已处理的论文结构"""
    metadata: PaperMetadata
    content_preview: str
    structured_analysis: Dict[str, Any]
    references: List[Dict[str, str]]
    concepts: List[str]
    processed_at: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "metadata": self.metadata.to_dict(),
            "content_preview": self.content_preview,
            "structured_analysis": self.structured_analysis,
            "references": self.references,
            "concepts": self.concepts,
            "processed_at": self.processed_at
        }


@dataclass
class KnowledgeUnit:
    """知识单元结构"""
    concept: str
    definition: str
    sources: List[str]  # 来源论文的arxiv_id
    relationships: List[Dict[str, Any]]  # 与其他概念的关系
    confidence: float  # 置信度
    created_at: str

    def to_markdown(self) -> str:
        """转换为Markdown格式"""
        md = f"## {self.concept}\n\n"
        md += f"**定义**: {self.definition}\n\n"
        md += f"**置信度**: {self.confidence:.2f}\n\n"

        if self.sources:
            md += "**来源**:\n"
            for source in self.sources:
                md += f"- {source}\n"
            md += "\n"

        if self.relationships:
            md += "**相关概念**:\n"
            for rel in self.relationships:
                md += f"- {rel['target']} ({rel['relationship']})\n"

        return md


class AgentState(TypedDict):
    """
    Agent状态定义
    核心理念：状态是Agent的完整工作上下文
    """
    # 基础信息
    domain: str  # 研究领域
    step: str  # 当前步骤
    error: Optional[str]  # 错误信息
    start_time: str  # 开始时间

    # 论文处理状态
    papers_found: List[PaperMetadata]  # 找到的论文
    papers_selected: List[PaperMetadata]  # 选中的论文
    papers_processed: List[ProcessedPaper]  # 已处理的论文
    current_paper_index: int  # 当前处理的论文索引

    # 知识构建状态
    knowledge_units: List[KnowledgeUnit]  # 知识单元
    concept_graph: Dict[str, Any]  # 概念图
    structured_knowledge: Dict[str, Any]  # 结构化知识

    # 对话和记忆
    messages: List[Dict[str, Any]]  # 对话历史
    reasoning_traces: List[Dict[str, Any]]  # 推理轨迹
    decisions: List[Dict[str, Any]]  # 关键决策记录

    # 性能指标
    tokens_used: int  # 已使用的token数
    papers_downloaded: int  # 已下载的论文数
    processing_time: float  # 处理时间


class LearningStateManager:
    """学习状态管理器"""

    def __init__(self, domain: str):
        self.state: AgentState = self._create_initial_state(domain)
        self.state_history: List[AgentState] = []

    def _create_initial_state(self, domain: str) -> AgentState:
        """创建初始状态"""
        return {
            "domain": domain,
            "step": "initialized",
            "error": None,
            "start_time": datetime.now().isoformat(),

            "papers_found": [],
            "papers_selected": [],
            "papers_processed": [],
            "current_paper_index": 0,

            "knowledge_units": [],
            "concept_graph": {"nodes": [], "edges": []},
            "structured_knowledge": {},

            "messages": [],
            "reasoning_traces": [],
            "decisions": [],

            "tokens_used": 0,
            "papers_downloaded": 0,
            "processing_time": 0.0
        }

    def update_state(self, **updates):
        """更新状态"""
        for key, value in updates.items():
            if key in self.state:
                self.state[key] = value
            else:
                raise KeyError(f"Invalid state key: {key}")

        # 记录状态历史
        self.state_history.append(self.state.copy())

    def add_paper(self, paper: PaperMetadata):
        """添加论文"""
        self.state["papers_found"].append(paper)

    def add_processed_paper(self, paper: ProcessedPaper):
        """添加已处理论文"""
        self.state["papers_processed"].append(paper)

    def add_knowledge_unit(self, unit: KnowledgeUnit):
        """添加知识单元"""
        self.state["knowledge_units"].append(unit)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息"""
        message = {"role": role, "content": content}
        if metadata:
            message["metadata"] = metadata
        self.state["messages"].append(message)

    def add_reasoning_trace(self, trace: Dict[str, Any]):
        """添加推理轨迹"""
        self.state["reasoning_traces"].append(trace)

    def record_decision(self, decision: str, rationale: str, context: Dict[str, Any]):
        """记录决策"""
        self.state["decisions"].append({
            "decision": decision,
            "rationale": rationale,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })

    def increment_tokens(self, tokens: int):
        """增加token计数"""
        self.state["tokens_used"] += tokens

    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "domain": self.state["domain"],
            "step": self.state["step"],
            "papers_found": len(self.state["papers_found"]),
            "papers_processed": len(self.state["papers_processed"]),
            "knowledge_units": len(self.state["knowledge_units"]),
            "tokens_used": self.state["tokens_used"],
            "has_error": self.state["error"] is not None
        }

    def save_state(self, filepath: str):
        """保存状态到文件"""
        # 转换为可序列化的字典
        state_dict = {
            "state": self._state_to_dict(self.state),
            "history": [self._state_to_dict(s) for s in self.state_history],
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)

    def _state_to_dict(self, state: AgentState) -> Dict[str, Any]:
        """将状态转换为字典"""
        result = {}
        for key, value in state.items():
            if isinstance(value, list):
                if value and isinstance(value[0], (PaperMetadata, ProcessedPaper, KnowledgeUnit)):
                    result[key] = [item.to_dict() if hasattr(item, 'to_dict') else item
                                   for item in value]
                else:
                    result[key] = value
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def load_state(cls, filepath: str) -> 'LearningStateManager':
        """从文件加载状态"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        manager = cls(data["state"]["domain"])
        manager.state = cls._dict_to_state(data["state"])
        manager.state_history = [cls._dict_to_state(s) for s in data.get("history", [])]

        return manager

    @classmethod
    def _dict_to_state(cls, data: Dict[str, Any]) -> AgentState:
        """从字典恢复状态"""
        state = {
            "domain": data["domain"],
            "step": data["step"],
            "error": data.get("error"),
            "start_time": data["start_time"],

            "papers_found": [PaperMetadata.from_dict(p) for p in data.get("papers_found", [])],
            "papers_selected": [PaperMetadata.from_dict(p) for p in data.get("papers_selected", [])],
            "papers_processed": [ProcessedPaper(**p) for p in data.get("papers_processed", [])],
            "current_paper_index": data.get("current_paper_index", 0),

            "knowledge_units": [KnowledgeUnit(**k) for k in data.get("knowledge_units", [])],
            "concept_graph": data.get("concept_graph", {"nodes": [], "edges": []}),
            "structured_knowledge": data.get("structured_knowledge", {}),

            "messages": data.get("messages", []),
            "reasoning_traces": data.get("reasoning_traces", []),
            "decisions": data.get("decisions", []),

            "tokens_used": data.get("tokens_used", 0),
            "papers_downloaded": data.get("papers_downloaded", 0),
            "processing_time": data.get("processing_time", 0.0)
        }

        return state