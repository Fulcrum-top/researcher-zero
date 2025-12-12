import os
import asyncio
import aiohttp
from typing import List, Dict, Optional
from langchain_core.tools import tool
from core.utils.logs import logger

SEMANTIC_SCHOLAR_DESCRIPTION = (
    "Search for academic papers on Semantic Scholar. "
    "Provides high-quality metadata including citation counts, authors, venues, and abstracts. "
    "Especially good for finding survey papers and high-impact research."
)


@tool(description=SEMANTIC_SCHOLAR_DESCRIPTION)
async def semantic_scholar_search(
        query: str,
        max_results: int = 20,
        year_filter: Optional[str] = None
) -> List[Dict]:
    """
    Search Semantic Scholar for academic papers.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        year_filter: Year filter in format "start_year-end_year"

    Returns:
        List of paper metadata dictionaries
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    # 构建查询参数
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,venue,abstract,citationCount,url,externalIds,openAccessPdf,referenceCount",
        "sort": "relevance"
    }

    # 添加时间过滤
    if year_filter:
        params["year"] = year_filter

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                    base_url,
                    params=params,
                    headers=headers,
                    timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return parse_semantic_scholar_results(data.get("data", []))
                elif response.status == 429:
                    logger.warning("Semantic Scholar rate limit exceeded")
                    return []
                else:
                    logger.error(f"Semantic Scholar API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []


def parse_semantic_scholar_results(results: List[Dict]) -> List[Dict]:
    """Parse Semantic Scholar API response into standardized format."""
    papers = []

    for item in results:
        try:
            # 提取作者
            authors = [author.get("name", "") for author in item.get("authors", [])]

            # 提取DOI
            doi = None
            external_ids = item.get("externalIds", {})
            if external_ids:
                doi = external_ids.get("DOI")

            # 提取PDF链接
            pdf_url = None
            open_access_pdf = item.get("openAccessPdf")
            if open_access_pdf:
                pdf_url = open_access_pdf.get("url")

            # 判断是否为综述论文
            title = item.get("title", "")
            abstract = item.get("abstract", "")
            is_survey = any(keyword in title.lower() or keyword in abstract.lower()
                            for keyword in ['survey', 'review', 'overview',
                                            'state of the art', 'literature review'])

            paper = {
                "title": title,
                "authors": authors,
                "abstract": abstract if abstract else "",
                "year": item.get("year", 0),
                "venue": item.get("venue", ""),
                "citation_count": item.get("citationCount", 0),
                "doi": doi,
                "url": item.get("url", ""),
                "pdf_url": pdf_url,
                "source": "semantic_scholar",
                "is_survey": is_survey,
                "source_id": item.get("paperId", ""),
                "relevance_score": 0.0,
                "citations": []
            }
            papers.append(paper)

        except Exception as e:
            logger.warning(f"Failed to parse Semantic Scholar result: {e}")
            continue

    logger.info(f"Semantic Scholar search returned {len(papers)} papers")
    return papers