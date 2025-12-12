import os
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from langchain_core.tools import tool
from core.utils.logs import logger

ARXIV_SEARCH_DESCRIPTION = (
    "Search for academic papers on arXiv pre-print server. "
    "Useful for finding the latest research papers in computer science, "
    "physics, mathematics, and other quantitative sciences."
)


@tool(description=ARXIV_SEARCH_DESCRIPTION)
async def arxiv_search(
        query: str,
        max_results: int = 20,
        category: Optional[str] = None
) -> List[Dict]:
    """
    Search arXiv for academic papers.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        category: arXiv category (e.g., cs.AI, cs.LG, stat.ML)

    Returns:
        List of paper metadata dictionaries
    """
    base_url = "http://export.arxiv.org/api/query"

    # 构建查询参数
    search_query = query
    if category:
        search_query = f"cat:{category} AND {query}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(base_url, params=params, timeout=30) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    return parse_arxiv_results(xml_content, max_results)
                else:
                    logger.error(f"arXiv API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []


def parse_arxiv_results(xml_content: str, max_results: int) -> List[Dict]:
    """Parse arXiv XML response into standardized format."""
    try:
        root = ET.fromstring(xml_content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}

        papers = []
        entries = root.findall('atom:entry', namespace)

        for entry in entries[:max_results]:
            try:
                # 提取标题
                title_elem = entry.find('atom:title', namespace)
                title = title_elem.text.strip() if title_elem is not None else ""

                # 提取摘要
                summary_elem = entry.find('atom:summary', namespace)
                abstract = summary_elem.text.strip() if summary_elem is not None else ""

                # 提取作者
                authors = []
                for author_elem in entry.findall('atom:author', namespace):
                    name_elem = author_elem.find('atom:name', namespace)
                    if name_elem is not None:
                        authors.append(name_elem.text)

                # 提取发布日期和年份
                published_elem = entry.find('atom:published', namespace)
                published = published_elem.text if published_elem is not None else ""
                year = int(published[:4]) if published and len(published) >= 4 else 0

                # 提取arXiv ID
                id_elem = entry.find('atom:id', namespace)
                arxiv_id = ""
                if id_elem is not None:
                    arxiv_id = id_elem.text.split('/')[-1]

                # 判断是否为综述论文
                is_survey = any(keyword in title.lower() or keyword in abstract.lower()
                                for keyword in ['survey', 'review', 'overview',
                                                'state of the art', 'literature review'])

                paper = {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "year": year,
                    "venue": "arXiv",
                    "citation_count": None,  # arXiv不提供引用数
                    "doi": None,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    "source": "arxiv",
                    "is_survey": is_survey,
                    "source_id": arxiv_id,
                    "relevance_score": 0.0,
                    "citations": []
                }
                papers.append(paper)

            except Exception as e:
                logger.warning(f"Failed to parse arXiv entry: {e}")
                continue

        logger.info(f"arXiv search returned {len(papers)} papers")
        return papers

    except Exception as e:
        logger.error(f"Failed to parse arXiv XML: {e}")
        return []