"""
论文引用抽取工具
支持从参考文献中提取、去重、排序
"""
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """引用信息"""
    authors: List[str]
    title: str
    year: Optional[str] = None
    venue: Optional[str] = None
    source: Optional[str] = None  # 来源论文
    count: int = 1  # 出现次数


class CitationExtractor:
    """引用抽取工具"""

    def __init__(self):
        # 引用格式正则表达式
        self.patterns = {
            "apa": r'([A-Z][a-zA-Z\s,&\.\-]+)\((\d{4})\)\.\s*([^\.]+[\.?!])',
            "ieee": r'\[\d+\]\s+([^"]+),\s*"([^"]+)",\s*([^,]+),\s*(\d{4})',
            "bibtex": r'@\w+\{[^,]+,([^}]+)\}',
            "simple": r'([^\(]+)\((\d{4})\)\.\s*(.+)'
        }

    def extract_from_text(self, text: str) -> List[Citation]:
        """从文本中提取引用"""
        citations = []

        # 尝试不同格式
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                try:
                    citation = self._parse_match(match, pattern_name)
                    if citation:
                        citations.append(citation)
                except Exception as e:
                    logger.debug(f"解析引用失败: {e}")
                    continue

        return citations

    def _parse_match(self, match: re.Match, pattern_name: str) -> Optional[Citation]:
        """解析匹配结果"""
        if pattern_name == "apa":
            authors = match.group(1).strip()
            year = match.group(2)
            title = match.group(3).strip()
            return Citation(
                authors=self._parse_authors(authors),
                title=title,
                year=year
            )
        elif pattern_name == "ieee":
            authors = match.group(1).strip()
            title = match.group(2).strip()
            venue = match.group(3).strip()
            year = match.group(4)
            return Citation(
                authors=self._parse_authors(authors),
                title=title,
                year=year,
                venue=venue
            )
        elif pattern_name == "simple":
            authors = match.group(1).strip()
            year = match.group(2)
            title = match.group(3).strip()
            return Citation(
                authors=self._parse_authors(authors),
                title=title,
                year=year
            )

        return None

    def _parse_authors(self, author_str: str) -> List[str]:
        """解析作者字符串"""
        authors = []

        # 分割作者
        if ' and ' in author_str.lower():
            parts = re.split(r'\sand\s', author_str, flags=re.IGNORECASE)
        elif ' & ' in author_str:
            parts = author_str.split(' & ')
        elif ',' in author_str:
            parts = author_str.split(',')
        else:
            parts = [author_str]

        # 清理每个作者
        for part in parts:
            author = part.strip()
            if author and len(author) > 1:
                authors.append(author)

        return authors[:10]  # 限制作者数量

    def deduplicate_and_rank(
            self,
            citations: List[Citation],
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        去重并排序引用

        Args:
            citations: 原始引用列表
            top_k: 返回的Top K引用

        Returns:
            排序后的引用列表
        """
        # 创建引用键用于去重
        citation_keys = {}

        for citation in citations:
            # 创建唯一键（作者+年份+标题前20字符）
            author_part = "".join(citation.authors[:2]) if citation.authors else ""
            year_part = citation.year or ""
            title_part = citation.title[:20] if citation.title else ""
            key = f"{author_part}{year_part}{title_part}".lower().replace(" ", "")

            if key in citation_keys:
                # 合并引用，增加计数
                existing = citation_keys[key]
                existing.count += 1
                # 补充缺失的信息
                if not existing.venue and citation.venue:
                    existing.venue = citation.venue
                if not existing.year and citation.year:
                    existing.year = citation.year
            else:
                citation_keys[key] = citation

        # 排序（按出现次数降序）
        sorted_citations = sorted(
            citation_keys.values(),
            key=lambda x: x.count,
            reverse=True
        )

        # 转换为字典格式
        result = []
        for i, citation in enumerate(sorted_citations[:top_k]):
            result.append({
                "rank": i + 1,
                "count": citation.count,
                "authors": citation.authors,
                "title": citation.title,
                "year": citation.year,
                "venue": citation.venue,
                "source": citation.source
            })

        return result

    async def extract_references(
            self,
            papers_info: List[Dict[str, Any]],
            top_k: int = 10
    ) -> Dict[str, Any]:
        """
        从多篇论文中提取引用

        Args:
            papers_info: 论文信息列表
            top_k: 返回的Top K引用

        Returns:
            引用分析结果
        """
        all_citations = []

        for paper in papers_info:
            # 从论文内容中提取引用
            content = paper.get("content", "")
            references_section = paper.get("references_section", "")

            # 优先使用references_section
            text_to_analyze = references_section if references_section else content

            citations = self.extract_from_text(text_to_analyze)

            # 标记来源
            for citation in citations:
                citation.source = paper.get("title", "Unknown")

            all_citations.extend(citations)

        # 去重排序
        ranked_citations = self.deduplicate_and_rank(all_citations, top_k)

        # 分析引用模式
        citation_analysis = self._analyze_citation_patterns(ranked_citations)

        return {
            "total_citations_found": len(all_citations),
            "unique_citations": len(ranked_citations),
            "top_citations": ranked_citations,
            "analysis": citation_analysis
        }

    def _analyze_citation_patterns(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析引用模式"""
        if not citations:
            return {}

        # 分析年份分布
        years = [c["year"] for c in citations if c["year"]]
        year_counts = Counter(years)

        # 分析高频作者
        all_authors = []
        for citation in citations:
            all_authors.extend(citation.get("authors", []))
        author_counts = Counter(all_authors)

        # 分析研究领域（通过高频词）
        titles = " ".join([c["title"].lower() for c in citations if c["title"]])
        words = re.findall(r'\b[a-z]{4,}\b', titles)
        word_counts = Counter(words)

        return {
            "most_cited_years": year_counts.most_common(5),
            "most_frequent_authors": author_counts.most_common(10),
            "common_keywords": word_counts.most_common(20)
        }


# MCP工具包装器
class CitationToolMCPWrapper:
    """引用工具的MCP包装器"""

    def __init__(self):
        self.tool = CitationExtractor()

    def get_tools(self) -> List[Dict[str, Any]]:
        """返回MCP工具定义"""
        return [
            {
                "name": "extract_references",
                "description": "从多篇论文中提取引用并进行去重排序",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "papers_info": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "references_section": {"type": "string"}
                                }
                            },
                            "description": "论文信息列表"
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 10,
                            "description": "返回的Top K引用数量"
                        }
                    },
                    "required": ["papers_info"]
                }
            }
        ]

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        if tool_name == "extract_references":
            return await self.tool.extract_references(**kwargs)
        else:
            raise ValueError(f"未知工具: {tool_name}")
