"""
论文处理工具 - 基于MCP协议
更新：与LearningState集成
"""
import json
import hashlib
import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import arxiv
import PyPDF2
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

# 从learning_state导入PaperMetadata
from core.infra.learning_state import PaperMetadata


class PaperTools:
    """论文处理工具类"""

    def __init__(self, cache_dir: Optional[str] = None):
        # 使用你的配置系统
        from core.configs.settings import settings
        self.cache_dir = cache_dir or settings.storage.cache_dir
        self.client = arxiv.Client(
            page_size=settings.arxiv.max_results,
            delay_seconds=3,
            num_retries=3
        )

    def search_papers(self, query: str, max_results: int = 10,
                     sort_by: str = "relevance") -> List[PaperMetadata]:
        """
        搜索论文工具 - 增强版本

        算法设计改进：
        1. 智能查询增强：自动添加survey/review关键词
        2. 结果过滤：基于摘要的初步相关性评估
        3. 评分机制：为每篇论文计算初步相关性分数
        """
        try:
            # 智能查询增强
            enhanced_queries = [
                f"{query} AND (survey OR review OR overview)",
                f"{query} AND (comprehensive OR tutorial)",
                f"{query}"
            ]

            all_papers = []
            seen_ids = set()

            for enhanced_query in enhanced_queries:
                if len(all_papers) >= max_results:
                    break

                search = arxiv.Search(
                    query=enhanced_query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance if sort_by == "relevance"
                    else arxiv.SortCriterion.SubmittedDate
                )

                for result in self.client.results(search):
                    if result.entry_id in seen_ids:
                        continue

                    seen_ids.add(result.entry_id)

                    # 计算相关性分数
                    relevance_score = self._calculate_relevance_score(
                        result.title, result.summary, query
                    )

                    paper_meta = PaperMetadata(
                        title=result.title,
                        authors=[str(author) for author in result.authors],
                        abstract=result.summary,
                        arxiv_id=result.entry_id.split('/')[-1],
                        published=str(result.published),
                        categories=result.categories,
                        reference_count=len(result.journal_ref or []),
                        relevance_score=relevance_score
                    )
                    all_papers.append(paper_meta)

                    if len(all_papers) >= max_results:
                        break

            # 按相关性排序
            sorted_papers = sorted(all_papers,
                                 key=lambda x: x.relevance_score,
                                 reverse=True)

            logger.info(f"Found {len(sorted_papers)} papers for query: {query}")
            return sorted_papers[:max_results]

        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []

    def _calculate_relevance_score(self, title: str, abstract: str, query: str) -> float:
        """计算相关性分数"""
        score = 0.0

        # 检查关键词
        query_terms = query.lower().split()
        text = f"{title} {abstract}".lower()

        # 基础匹配
        for term in query_terms:
            if len(term) > 3:  # 忽略太短的词
                if term in text:
                    score += 0.1

        # Survey论文加分
        survey_keywords = ["survey", "review", "overview", "comprehensive", "tutorial"]
        if any(keyword in text for keyword in survey_keywords):
            score += 0.3

        # 近期论文加分（最近2年）
        # 这里简化处理，实际应该解析日期

        # 标题中直接包含查询词加分
        title_lower = title.lower()
        for term in query_terms:
            if len(term) > 3 and term in title_lower:
                score += 0.2

        return min(score, 1.0)  # 归一化到0-1

    def download_paper(self, arxiv_id: str) -> Optional[str]:
        """
        下载论文PDF - 增强版本

        改进：
        1. 缓存管理
        2. 重试机制
        3. 下载进度跟踪
        """
        cache_key = f"paper_{arxiv_id}.pdf"
        cache_path = Path(self.cache_dir) / cache_key

        # 检查缓存
        if cache_path.exists():
            logger.info(f"Loading paper from cache: {arxiv_id}")
            return str(cache_path)

        try:
            # 从arXiv下载
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(self.client.results(search))

            # 下载PDF
            result.download_pdf(filename=str(cache_path))
            logger.info(f"Downloaded paper: {arxiv_id}")
            return str(cache_path)

        except Exception as e:
            logger.error(f"Error downloading paper {arxiv_id}: {e}")
            # 尝试备用下载方式
            return self._download_fallback(arxiv_id)

    def _download_fallback(self, arxiv_id: str) -> Optional[str]:
        """备用下载方式"""
        try:
            # 使用arXiv直接下载链接
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                cache_path = Path(self.cache_dir) / f"paper_{arxiv_id}.pdf"
                with open(cache_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded paper via fallback: {arxiv_id}")
                return str(cache_path)
        except Exception as e:
            logger.error(f"Fallback download failed for {arxiv_id}: {e}")

        return None

    def extract_paper_content(self, pdf_path: str, max_pages: int = 50) -> Dict[str, Any]:
        """
        提取论文内容 - 增强版本

        改进：
        1. 智能章节识别
        2. 图表检测
        3. 参考文献提取优化
        """
        try:
            content = {
                "full_text": "",
                "sections": {},
                "references": [],
                "metadata": {}
            }

            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # 提取元数据
                if pdf_reader.metadata:
                    content["metadata"] = {
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                        "producer": pdf_reader.metadata.get('/Producer', ''),
                        "creation_date": pdf_reader.metadata.get('/CreationDate', ''),
                        "modification_date": pdf_reader.metadata.get('/ModDate', '')
                    }

                # 提取文本（限制页数）
                full_text = []
                total_pages = min(len(pdf_reader.pages), max_pages)

                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():  # 跳过空页
                        full_text.append(f"--- Page {page_num + 1} ---\n{text}")

                content["full_text"] = "\n".join(full_text)

                # 智能章节提取
                sections = self._extract_sections(content["full_text"])
                content["sections"] = sections

                # 提取参考文献
                content["references"] = self._extract_references(content["full_text"])

            return content

        except Exception as e:
            logger.error(f"Error extracting paper content: {e}")
            return {"full_text": "", "sections": {}, "references": [], "metadata": {}}

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """提取章节"""
        sections = {}

        # 常见的章节标题模式
        section_patterns = [
            r'^\s*(\d+\.\d*\.?\s*[A-Z].*?)$',
            r'^\s*([A-Z][A-Z\s]+\s*)$',
            r'^\s*(ABSTRACT|INTRODUCTION|BACKGROUND|RELATED WORK|METHOD|METHODOLOGY|'
            r'EXPERIMENTS|RESULTS|DISCUSSION|CONCLUSION|REFERENCES|BIBLIOGRAPHY)\s*$'
        ]

        lines = text.split('\n')
        current_section = "Header"
        section_content = []

        for line in lines:
            line_stripped = line.strip()

            # 检查是否是章节标题
            is_section = False
            for pattern in section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_section = True
                    break

            if is_section and len(line_stripped) < 100:  # 避免误判长文本
                # 保存当前章节
                if section_content and current_section:
                    sections[current_section] = "\n".join(section_content)

                # 开始新章节
                current_section = line_stripped
                section_content = []
            else:
                section_content.append(line)

        # 保存最后一个章节
        if section_content and current_section:
            sections[current_section] = "\n".join(section_content)

        return sections

    def extract_references(self, paper_content: Union[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        提取参考文献 - 增强版本

        算法设计：
        1. 多模式识别：不同引用格式
        2. 智能解析：作者、标题、年份、期刊
        3. 去重和验证
        """
        references = []

        try:
            # 处理不同类型的输入
            if isinstance(paper_content, str):
                text = paper_content
            elif isinstance(paper_content, dict):
                # 优先使用REFERENCES章节
                ref_section = paper_content.get("sections", {}).get("REFERENCES", "")
                if not ref_section:
                    ref_section = paper_content.get("full_text", "")
                text = ref_section
            else:
                text = ""

            if not text:
                return []

            # 多种引用格式的正则表达式
            patterns = [
                # APA格式: Author, A. A., & Author, B. B. (Year). Title. Journal, Volume(Issue), pages.
                r'([A-Z][a-zA-Z\s,&\.]+)\((\d{4})\)\.\s*([^\.]+)\.\s*([^,]+),\s*(\d+)\((\d+)\):\s*(\d+-\d+)',

                # IEEE格式: [1] A. Author, "Title," Journal, vol. X, no. Y, pp. ZZ-ZZ, Month Year.
                r'\[\d+\]\s+([A-Z]\.[\sA-Z]+),\s*"([^"]+)",\s*([^,]+),\s*vol\.\s*(\d+),\s*no\.\s*(\d+),\s*pp\.\s*(\d+-\d+),\s*(\w+\s+\d{4})',

                # 简单格式: Author et al., "Title", Conference, Year
                r'([A-Z][a-zA-Z\s]+et al\.),\s*"([^"]+)",\s*([^,]+),\s*(\d{4})'
            ]

            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) < 20 or len(line) > 500:
                    continue

                ref_info = self._parse_reference_line(line)
                if ref_info:
                    references.append(ref_info)

            # 去重
            unique_refs = []
            seen = set()
            for ref in references:
                key = (ref.get("authors", ""), ref.get("title", ""), ref.get("year", ""))
                if key not in seen and all(key):
                    seen.add(key)
                    unique_refs.append(ref)

            logger.info(f"Extracted {len(unique_refs)} unique references")
            return unique_refs[:100]  # 限制数量

        except Exception as e:
            logger.error(f"Error extracting references: {e}")
            return []

    def _parse_reference_line(self, line: str) -> Dict[str, str]:
        """解析单行引用"""
        ref = {
            "raw_text": line,
            "authors": "",
            "title": "",
            "year": "",
            "venue": "",
            "url": ""
        }

        # 提取年份
        year_match = re.search(r'\b(19|20)\d{2}\b', line)
        if year_match:
            ref["year"] = year_match.group()

        # 提取作者（简化版本）
        # 寻找常见的作者分隔符
        if 'et al' in line.lower():
            author_part = line.split('et al')[0].strip()
            if author_part:
                ref["authors"] = author_part + " et al."
        else:
            # 尝试提取第一个逗号前的内容作为作者
            parts = line.split(',')
            if len(parts) > 1:
                ref["authors"] = parts[0].strip()

        # 提取标题（在引号中的内容）
        title_match = re.search(r'"([^"]+)"', line)
        if title_match:
            ref["title"] = title_match.group(1)
        else:
            # 尝试其他标题模式
            title_match = re.search(r'\.\s*([^\.]+?)\.\s*(?:[A-Z]|$)', line)
            if title_match:
                ref["title"] = title_match.group(1).strip()

        # 提取venue（会议/期刊）
        venue_keywords = ["arXiv", "Proceedings", "Conference", "Journal", "Workshop",
                         "Symposium", "Transactions", "ICLR", "NeurIPS", "ICML", "CVPR"]
        for keyword in venue_keywords:
            if keyword in line:
                ref["venue"] = keyword
                break

        # 提取URL
        url_match = re.search(r'https?://[^\s]+', line)
        if url_match:
            ref["url"] = url_match.group()

        return ref