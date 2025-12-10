"""
论文搜索工具实现
基于MCP协议，提供智能论文搜索功能
"""
import arxiv
import re
from typing import List, Dict, Any, Optional
from core.mcp.models import SearchMode


class PaperSearchTool:
    """论文搜索工具"""

    def __init__(self):
        self.client = arxiv.Client(page_size=100, delay_seconds=3.0)

    async def search_papers(
        self,
        query: str,
        mode: str = SearchMode.GENERAL,
        category: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索论文的核心逻辑
        
        Args:
            query: 搜索关键词
            mode: 搜索模式 ("survey" 或 "general")
            category: 学科分类 (如 "cs.AI")
            max_results: 最大返回结果数
            
        Returns:
            论文信息列表
        """
        # 构建查询
        if mode == SearchMode.SURVEY:
            arxiv_query = self._build_survey_query(query, category)
        else:
            arxiv_query = self._build_general_query(query, category)

        # 执行搜索
        search = arxiv.Search(
            query=arxiv_query,
            max_results=min(max_results, 100),
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in self.client.results(search):
            result = {
                "title": paper.title,
                "authors": [str(author) for author in paper.authors],
                "abstract": paper.summary,
                "arxiv_id": paper.entry_id.split('/')[-1],
                "published": paper.published.isoformat() if paper.published else "",
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "relevance_score": getattr(paper, '_raw_score', 0.0)
            }

            # Survey模式额外计算置信度
            if mode == SearchMode.SURVEY:
                result["is_survey_candidate"] = self._is_survey_paper(paper.title, paper.summary)
                result["survey_confidence"] = self._calculate_survey_confidence(paper.title, paper.summary)

            results.append(result)

            if len(results) >= max_results:
                break

        return results

    def _build_survey_query(self, base_query: str, category: Optional[str]) -> str:
        """构建Survey查询"""
        cleaned = re.sub(r'[{}]', '', base_query).strip()
        survey_keywords = ["survey", "review", "overview", "tutorial", "comprehensive"]

        title_query = " OR ".join([f'ti:"{kw}"' for kw in survey_keywords[:3]])
        abstract_query = " OR ".join([f'abs:"{kw}"' for kw in survey_keywords[:3]])

        query = f'({cleaned}) AND (({title_query}) OR ({abstract_query}))'

        if category:
            query += f' AND cat:{category}'

        return query

    def _build_general_query(self, base_query: str, category: Optional[str]) -> str:
        """构建普通查询"""
        cleaned = re.sub(r'[{}]', '', base_query).strip()

        if category:
            return f'({cleaned}) AND cat:{category}'

        return cleaned

    def _is_survey_paper(self, title: str, abstract: str) -> bool:
        """判断是否为Survey论文"""
        text = f"{title} {abstract}".lower()
        indicators = ["survey", "review", "overview", "tutorial", "comprehensive"]
        return any(indicator in text for indicator in indicators)

    def _calculate_survey_confidence(self, title: str, abstract: str) -> float:
        """计算Survey置信度"""
        text = f"{title} {abstract}".lower()
        confidence = 0.0

        if "a survey of" in text or "a review of" in text:
            confidence += 0.4

        for keyword in ["survey", "review", "overview"]:
            if keyword in text:
                confidence += 0.2
            if keyword in title.lower():
                confidence += 0.1

        return min(confidence, 1.0)


# MCP工具包装器
class PaperSearchMCPWrapper:
    """论文搜索工具的MCP包装器"""

    def __init__(self):
        self.tool = PaperSearchTool()

    def get_tools(self) -> List[Dict[str, Any]]:
        """返回MCP工具定义"""
        return [
            {
                "name": "search_papers",
                "description": "搜索学术论文，支持Survey模式和普通模式。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "mode": {
                            "type": "string", 
                            "enum": [SearchMode.SURVEY, SearchMode.GENERAL], 
                            "default": SearchMode.GENERAL,
                            "description": "搜索模式: survey(查找综述论文) 或 general(一般搜索)"
                        },
                        "category": {
                            "type": "string", 
                            "description": "学科分类，如 cs.AI, cs.LG 等"
                        },
                        "max_results": {
                            "type": "integer", 
                            "default": 10,
                            "description": "最大返回结果数"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        if tool_name == "search_papers":
            return await self.tool.search_papers(**kwargs)
        else:
            raise ValueError(f"未知工具: {tool_name}")