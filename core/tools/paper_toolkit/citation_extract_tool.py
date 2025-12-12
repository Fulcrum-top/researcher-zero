# core/tools/paper_toolkit/citation_extract_tool.py
"""
论文引用抽取与核心文献识别工具
严格按照规则实现：
1. 引用抽取工具：输入单篇论文（web链接），输出该论文的引用文献列表
2. 核心文献识别工具：输入多个引用列表，返回topk被引文献按频率降序列表
"""

import os
import json
import asyncio
import re
from typing import List, Optional, Dict, Any, TypedDict
from dataclasses import dataclass
from collections import Counter
from datetime import datetime
import logging

from langchain_core.tools import tool
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# 设置日志
from utils.logs import logger


# ==================== 数据模型 ====================

class CitationExtractionResult(BaseModel):
    """引用抽取结果模型"""
    citations: List[str] = Field(
        description="论文的引用文献标题列表"
    )
    source_paper: str = Field(
        description="源论文标识符"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )


class CorePapersResult(BaseModel):
    """核心文献识别结果模型"""
    top_citations: List[str] = Field(
        description="按频率降序排列的topk被引文献标题列表"
    )
    frequencies: Dict[str, int] = Field(
        description="每个文献的引用频率"
    )
    top_k: int = Field(
        description="返回的topk值"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )


# ==================== arXiv引用抽取工具 ====================

class ArxivCitationExtractor:
    """arXiv论文引用抽取器"""

    BASE_URL = "https://arxiv.org"

    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def extract_citations_from_arxiv_link(self, arxiv_link: str) -> List[str]:
        """
        从arXiv链接提取引用列表

        Args:
            arxiv_link: arXiv论文链接，如"https://arxiv.org/abs/1706.03762"

        Returns:
            引用文献标题列表
        """
        try:
            # 从链接中提取arXiv ID
            arxiv_id = self._extract_arxiv_id_from_link(arxiv_link)
            if not arxiv_id:
                logger.error(f"无法从链接中提取arXiv ID: {arxiv_link}")
                return []

            # 构建arXiv页面URL
            url = f"{self.BASE_URL}/abs/{arxiv_id}"

            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    citations = self._parse_citations_from_html(html)
                    logger.info(f"从 {arxiv_link} 成功提取 {len(citations)} 个引用")
                    return citations
                else:
                    logger.error(f"无法访问arXiv页面: {url}, 状态码: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"从arXiv链接提取引用失败: {e}")
            return []

    def _extract_arxiv_id_from_link(self, link: str) -> Optional[str]:
        """从链接中提取arXiv ID"""
        # 支持的arXiv链接格式
        patterns = [
            r'arxiv\.org/abs/([\d\.]+v?\d*)',
            r'arxiv\.org/pdf/([\d\.]+v?\d*)',
            r'arxiv\.org/html/([\d\.]+v?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, link)
            if match:
                return match.group(1)

        # 如果没有匹配，尝试直接作为ID
        if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', link):
            return link

        return None

    def _parse_citations_from_html(self, html: str) -> List[str]:
        """
        从HTML页面解析引用列表，只返回标题

        Args:
            html: HTML内容

        Returns:
            引用文献标题列表
        """
        soup = BeautifulSoup(html, 'html.parser')

        # 查找引用部分的策略
        citations = []

        # 策略1：查找引用部分的标题
        reference_sections = self._find_reference_sections(soup)
        for section in reference_sections:
            section_citations = self._extract_titles_from_reference_section(section)
            citations.extend(section_citations)

        # 如果找到了引用，返回
        if citations:
            # 去重并清理
            unique_citations = []
            seen = set()
            for citation in citations:
                if citation and citation not in seen:
                    seen.add(citation)
                    unique_citations.append(citation)
            return unique_citations

        # 策略2：查找所有arXiv链接的标题
        arxiv_citations = self._extract_titles_from_arxiv_links(soup)
        if arxiv_citations:
            return arxiv_citations

        # 策略3：通用模式匹配
        generic_citations = self._extract_titles_generic(html)
        return generic_citations

    def _find_reference_sections(self, soup) -> List:
        """查找引用部分"""
        sections = []

        # 可能的引用部分标题
        reference_headers = [
            "References",
            "Bibliography",
            "References and Notes",
            "REFERENCES",
            "BIBLIOGRAPHY"
        ]

        for header in reference_headers:
            # 查找包含标题的元素
            header_elem = soup.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b'],
                                    string=lambda text: text and header.lower() in text.lower())

            if header_elem:
                # 查找标题后的内容
                next_elem = header_elem.find_next(['ol', 'ul', 'div', 'p'])
                if next_elem:
                    sections.append(next_elem)

        return sections

    def _extract_titles_from_reference_section(self, section) -> List[str]:
        """从引用部分提取标题"""
        titles = []

        # 如果是列表
        if section.name in ['ol', 'ul']:
            items = section.find_all('li')
        else:
            # 尝试按行分割
            text = section.get_text()
            items = re.split(r'\n\s*\d+\.\s+|\n\s*\[\d+\]\s+', text)
            if len(items) > 1:
                items = items[1:]  # 跳过第一个空串或标题

        for item in items:
            if hasattr(item, 'get_text'):
                item_text = item.get_text().strip()
            else:
                item_text = str(item).strip()

            if item_text:
                title = self._extract_title_from_citation_text(item_text)
                if title:
                    titles.append(title)

        return titles

    def _extract_title_from_citation_text(self, text: str) -> Optional[str]:
        """从引用文本中提取标题"""
        # 常见的标题提取模式
        patterns = [
            # 双引号包围的标题
            r'"([^"]+)"',
            # 书名号包围的标题
            r'《([^》]+)》',
            # 斜体标记的标题
            r'<i>([^<]+)</i>',
            r'\*([^*]+)\*',
            # 论文标题通常出现在作者之后，期刊之前
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\.\s+"?([^"]+)"?\.',
            # arXiv标题
            r'arXiv:\s*[\d\.]+v?\d*\s*[^\.]*\.?\s*"([^"]+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                title = match.group(1).strip()
                if len(title) > 10:  # 太短的可能是错误匹配
                    return title

        # 如果没有匹配到模式，尝试提取看起来像标题的部分
        # 通常标题在第一个句号之前，且包含多个单词
        first_sentence = text.split('.')[0]
        words = first_sentence.split()
        if 3 <= len(words) <= 20:  # 合理的标题长度
            return first_sentence

        return None

    def _extract_titles_from_arxiv_links(self, soup) -> List[str]:
        """从arXiv链接中提取标题"""
        titles = []

        # 查找所有arXiv链接
        arxiv_links = soup.find_all('a', href=lambda href: href and 'arxiv.org/abs/' in href)

        for link in arxiv_links:
            # 尝试从链接文本获取
            link_text = link.get_text().strip()
            if link_text and len(link_text) > 10:  # 太短的可能是编号
                titles.append(link_text)
            else:
                # 从href中提取arXiv ID作为备选
                href = link.get('href', '')
                match = re.search(r'arxiv\.org/abs/([\d\.]+v?\d*)', href)
                if match:
                    titles.append(f"arXiv:{match.group(1)}")

        return titles

    def _extract_titles_generic(self, html: str) -> List[str]:
        """通用模式匹配提取标题"""
        titles = []

        # 查找看起来像引用的文本块
        # 常见的引用特征：包含作者、年份、标题
        patterns = [
            r'"([^"]+)"\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d{4}',  # 标题在双引号内
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+et al\.\s+\d{4}\s+"([^"]+)"',  # et al. 格式
            r'In:\s*[^\.]+\.\s*\d{4}\s+"([^"]+)"',  # 会议/期刊格式
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, html, re.IGNORECASE | re.DOTALL)
            for match in matches:
                title = match.group(1).strip()
                if title and len(title) > 10:
                    titles.append(title)

        return list(set(titles))  # 去重


# ==================== 工具1: 引用抽取工具 ====================

@tool
async def extract_paper_citations(
        paper_link: str
) -> Dict:
    """
    从单篇论文中提取引用文献列表。

    输入应该是论文的web链接，目前主要支持arXiv链接。
    输出是该论文引用的所有文献的标题列表。

    Args:
        paper_link: 论文的web链接，例如"https://arxiv.org/abs/1706.03762"

    Returns:
        包含引用列表的字典，格式为{
            "citations": [引用标题1, 引用标题2, ...],
            "source_paper": "输入的论文链接",
            "metadata": {...}
        }
    """
    logger.info(f"开始从论文链接提取引用: {paper_link}")
    start_time = datetime.now()

    # 检查是否是arXiv链接
    if "arxiv.org" not in paper_link.lower():
        logger.warning(f"目前仅支持arXiv链接，收到: {paper_link}")
        result = CitationExtractionResult(
            citations=[],
            source_paper=paper_link,
            metadata={
                "error": "目前仅支持arXiv链接",
                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
            }
        )
        return result.dict()

    # 提取引用
    async with ArxivCitationExtractor() as extractor:
        citations = await extractor.extract_citations_from_arxiv_link(paper_link)

    processing_time = (datetime.now() - start_time).total_seconds()

    result = CitationExtractionResult(
        citations=citations,
        source_paper=paper_link,
        metadata={
            "total_citations_found": len(citations),
            "processing_time_seconds": processing_time,
            "source": "arxiv"
        }
    )

    logger.info(f"引用提取完成，找到 {len(citations)} 个引用，耗时: {processing_time:.2f}秒")
    return result.dict()


# ==================== 工具2: 核心文献识别工具 ====================

@tool
async def identify_top_citations(
        citations_lists: List[List[str]],
        top_k: int = 10
) -> Dict:
    """
    从多个引用列表中识别topk被引文献。

    输入是多个引用列表的列表，每个引用列表是一个字符串列表（文献标题）。
    基于引用频率进行统计，返回按频率降序排列的topk文献列表。

    Args:
        citations_lists: 多个引用列表的列表，例如[
            ["标题1", "标题2", "标题3"],
            ["标题2", "标题3", "标题4"],
            ...
        ]
        top_k: 返回排名前多少位的被引文献，默认为10

    Returns:
        包含topk文献列表的字典，格式为{
            "top_citations": [文献标题1, 文献标题2, ...],  # 按频率降序
            "frequencies": {"文献标题1": 频率1, "文献标题2": 频率2, ...},
            "top_k": 输入的top_k值,
            "metadata": {...}
        }
    """
    logger.info(f"开始识别核心文献，输入 {len(citations_lists)} 个引用列表，top_k={top_k}")
    start_time = datetime.now()

    try:
        # 验证输入
        if not citations_lists:
            logger.warning("输入为空列表")
            return CorePapersResult(
                top_citations=[],
                frequencies={},
                top_k=top_k,
                metadata={"error": "输入为空列表"}
            ).dict()

        # 统计所有引用
        all_citations = []
        for citation_list in citations_lists:
            if isinstance(citation_list, list):
                all_citations.extend(citation_list)
            else:
                logger.warning(f"跳过非列表元素: {type(citation_list)}")

        if not all_citations:
            logger.warning("没有找到有效的引用数据")
            return CorePapersResult(
                top_citations=[],
                frequencies={},
                top_k=top_k,
                metadata={"error": "没有找到有效的引用数据"}
            ).dict()

        # 频次统计
        freq_counter = Counter(all_citations)

        # 按频次降序排序
        sorted_items = freq_counter.most_common()

        # 应用TopK限制
        if top_k > 0:
            sorted_items = sorted_items[:top_k]

        # 提取标题和频率
        top_citations = [item[0] for item in sorted_items]
        frequencies = {item[0]: item[1] for item in sorted_items}

        # 计算统计信息
        total_citations = len(all_citations)
        unique_citations = len(freq_counter)
        processing_time = (datetime.now() - start_time).total_seconds()

        result = CorePapersResult(
            top_citations=top_citations,
            frequencies=frequencies,
            top_k=top_k,
            metadata={
                "total_citations_analyzed": total_citations,
                "unique_citations": unique_citations,
                "input_lists_count": len(citations_lists),
                "processing_time_seconds": processing_time
            }
        )

        logger.info(f"核心文献识别完成，找到 {len(top_citations)} 篇核心文献")
        return result.dict()

    except Exception as e:
        logger.error(f"核心文献识别失败: {e}")
        return CorePapersResult(
            top_citations=[],
            frequencies={},
            top_k=top_k,
            metadata={"error": str(e)}
        ).dict()


# ==================== 完整流程示例 ====================

async def complete_citation_analysis_workflow(
        paper_links: List[str],
        top_k: int = 10
) -> Dict:
    """
    完整流程示例：从多篇论文提取引用，然后识别核心文献

    Args:
        paper_links: 多篇论文的链接列表
        top_k: 返回排名前多少位的核心被引论文

    Returns:
        包含完整分析结果的字典
    """
    logger.info(f"开始完整引用分析流程，分析 {len(paper_links)} 篇论文")

    # 步骤1: 并行提取每篇论文的引用
    extraction_tasks = []
    for link in paper_links:
        task = asyncio.create_task(extract_paper_citations.ainvoke({"paper_link": link}))
        extraction_tasks.append((link, task))

    # 收集结果
    all_citations_lists = []
    extraction_results = {}

    for link, task in extraction_tasks:
        try:
            result = await task
            citations = result.get("citations", [])
            all_citations_lists.append(citations)
            extraction_results[link] = {
                "citations_count": len(citations),
                "success": True
            }
            logger.info(f"成功提取 {link} 的 {len(citations)} 个引用")
        except Exception as e:
            logger.error(f"提取 {link} 的引用失败: {e}")
            extraction_results[link] = {
                "citations_count": 0,
                "success": False,
                "error": str(e)
            }
            all_citations_lists.append([])  # 添加空列表以保持顺序

    # 步骤2: 识别核心文献
    if all_citations_lists:
        identification_result = await identify_top_citations.ainvoke({
            "citations_lists": all_citations_lists,
            "top_k": top_k
        })
    else:
        identification_result = {
            "top_citations": [],
            "frequencies": {},
            "top_k": top_k,
            "metadata": {"error": "没有成功提取到任何引用"}
        }

    # 返回完整结果
    return {
        "extraction_results": extraction_results,
        "identification_result": identification_result,
        "summary": f"分析了 {len(paper_links)} 篇论文，提取了 {sum(len(lst) for lst in all_citations_lists)} 个引用，识别出 {len(identification_result.get('top_citations', []))} 篇核心文献"
    }


# ==================== 测试代码 ====================

async def _test_citation_tools():
    """测试引用工具"""

    # 测试用例
    test_paper_link = "https://arxiv.org/abs/1706.03762"  # Attention Is All You Need

    logger.info("测试工具1: 引用抽取工具...")

    try:
        # 测试工具1
        extraction_result = await extract_paper_citations.ainvoke({
            "paper_link": test_paper_link
        })

        print("引用抽取结果:")
        print(f"源论文: {extraction_result['source_paper']}")
        print(f"找到 {len(extraction_result['citations'])} 个引用")
        print("前5个引用:")
        for i, citation in enumerate(extraction_result['citations'][:5], 1):
            print(f"  {i}. {citation}")

        # 测试工具2
        logger.info("测试工具2: 核心文献识别工具...")

        # 模拟多个引用列表
        sample_citations_lists = [
            ["Attention Is All You Need", "BERT", "GPT-3", "Transformer"],
            ["Attention Is All You Need", "BERT", "ResNet", "AlexNet"],
            ["GPT-3", "BERT", "T5", "BERT"],
            ["Attention Is All You Need", "Transformer", "BERT", "ResNet"],
        ]

        identification_result = await identify_top_citations.ainvoke({
            "citations_lists": sample_citations_lists,
            "top_k": 5
        })

        print("\n核心文献识别结果:")
        print(f"Top {identification_result['top_k']} 被引文献:")
        for i, citation in enumerate(identification_result['top_citations'], 1):
            freq = identification_result['frequencies'][citation]
            print(f"  {i}. {citation} (被引 {freq} 次)")

        # 测试完整流程
        logger.info("测试完整流程...")
        paper_links = [
            "https://arxiv.org/abs/1706.03762",  # Attention Is All You Need
            "https://arxiv.org/abs/1810.04805",  # BERT
        ]

        full_result = await complete_citation_analysis_workflow(
            paper_links=paper_links,
            top_k=5
        )

        print("\n完整流程结果摘要:")
        print(full_result["summary"])

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(_test_citation_tools())