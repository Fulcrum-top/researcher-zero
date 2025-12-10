"""
搜索工具 - 基于MCP协议
设计理念：提供多样化的知识检索能力
"""
import requests
import json
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from core.configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果结构"""
    title: str
    url: str
    snippet: str
    source: str  # arxiv, google_scholar, semantic_scholar, etc.
    relevance_score: float


class SearchTools:
    """搜索工具类"""

    def __init__(self):
        self.semantic_scholar_api_key = None  # 可配置

    def deep_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        深度搜索工具

        背景：传统搜索只能获取表层信息
        目标：提供深度、多源的学术搜索能力

        算法设计：
        1. 多源搜索：arXiv + 学术搜索引擎
        2. 结果聚合和去重
        3. 相关性评分

        FutureWork：
        - 集成更多学术数据库
        - 使用LLM进行相关性重排序
        """
        results = []

        try:
            # 1. arXiv搜索
            arxiv_results = self._search_arxiv(query, max_results // 2)
            results.extend(arxiv_results)

            # 2. 语义学者搜索（如有API）
            if self.semantic_scholar_api_key:
                scholar_results = self._search_semantic_scholar(query, max_results // 2)
                results.extend(scholar_results)

            # 3. 去重和排序
            unique_results = self._deduplicate_results(results)
            sorted_results = sorted(unique_results,
                                    key=lambda x: x.relevance_score,
                                    reverse=True)[:max_results]

            logger.info(f"Deep search found {len(sorted_results)} results")
            return sorted_results

        except Exception as e:
            logger.error(f"Error in deep search: {e}")
            return []

    def _search_arxiv(self, query: str, max_results: int) -> List[SearchResult]:
        """搜索arXiv"""
        # 使用arxiv库（已在paper_tools中实现）
        # 这里简化为调用paper_tools
        from core.tools.paper_tools import PaperTools
        paper_tools = PaperTools()

        papers = paper_tools.search_papers(query, max_results)

        results = []
        for paper in papers:
            result = SearchResult(
                title=paper.title,
                url=f"https://arxiv.org/abs/{paper.arxiv_id}",
                snippet=paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                source="arxiv",
                relevance_score=0.8  # 简单评分
            )
            results.append(result)

        return results

    def _search_semantic_scholar(self, query: str, max_results: int) -> List[SearchResult]:
        """搜索语义学者"""
        # 实现语义学者API调用
        # 这里返回空列表作为占位
        return []

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """去重"""
        seen_titles = set()
        unique_results = []

        for result in results:
            normalized_title = result.title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_results.append(result)

        return unique_results

    def analyze_search_trends(self, domain: str, timeframe: str = "1y") -> Dict[str, Any]:
        """
        分析搜索趋势

        设计理念：不仅要找知识，还要理解知识的发展脉络
        """
        # 实现趋势分析逻辑
        return {
            "domain": domain,
            "timeframe": timeframe,
            "hot_topics": [],
            "rising_terms": [],
            "key_authors": []
        }


class SearchToolsMCPWrapper:
    """将SearchTools包装为MCP工具"""

    def __init__(self):
        self.tools = SearchTools()

    def get_tools(self):
        """返回MCP工具描述"""
        return [
            {
                "name": "deep_search",
                "description": "Deep search across multiple academic sources",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "analyze_search_trends",
                "description": "Analyze search trends in a specific domain",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "description": "Research domain"},
                        "timeframe": {"type": "string", "default": "1y",
                                      "description": "Timeframe (e.g., 1y, 6m)"}
                    },
                    "required": ["domain"]
                }
            }
        ]

    def execute_tool(self, tool_name: str, **kwargs):
        """执行工具"""
        if tool_name == "deep_search":
            return self.tools.deep_search(**kwargs)
        elif tool_name == "analyze_search_trends":
            return self.tools.analyze_search_trends(**kwargs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")