import os
import asyncio
import aiohttp
from typing import List, Dict
from langchain_core.tools import tool
from core.utils.logs import logger

CITATION_LIST_DESCRIPTION = (
    "Get citation list for a paper from Semantic Scholar. "
    "Returns titles of papers that cite the given paper."
)


@tool(description=CITATION_LIST_DESCRIPTION)
async def get_citation_list(
        paper_id: str,
        max_citations: int = 50,
        source: str = "semantic_scholar"
) -> List[Dict]:
    """
    Get citation list for a paper.

    Args:
        paper_id: Paper identifier
        max_citations: Maximum number of citations to return
        source: Data source (currently only supports "semantic_scholar")

    Returns:
        List of citation metadata dictionaries
    """
    if source != "semantic_scholar":
        logger.warning(f"Citation list only supports semantic_scholar, got {source}")
        return []

    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    params = {
        "limit": max_citations,
        "fields": "title,authors,year"
    }

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
                    citations = data.get("data", [])

                    # 提取引用论文的标题
                    citation_titles = []
                    for citation in citations:
                        cited_paper = citation.get("citingPaper", {})
                        title = cited_paper.get("title", "")
                        if title:
                            citation_titles.append({
                                "title": title,
                                "authors": [a.get("name", "") for a in cited_paper.get("authors", [])],
                                "year": cited_paper.get("year", 0)
                            })

                    logger.info(f"Retrieved {len(citation_titles)} citations for paper {paper_id}")
                    return citation_titles
                else:
                    logger.error(f"Citation API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Failed to get citation list: {e}")
            return []