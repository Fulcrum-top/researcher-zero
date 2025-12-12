import os
import asyncio
import aiohttp
from typing import Dict, Optional
from langchain_core.tools import tool
from core.utils.logs import logger

PAPER_METADATA_DESCRIPTION = (
    "Get detailed metadata for a specific paper using its ID. "
    "Supports multiple sources including arXiv and Semantic Scholar."
)


@tool(description=PAPER_METADATA_DESCRIPTION)
async def get_paper_metadata(
        paper_id: str,
        source: str = "semantic_scholar"
) -> Dict:
    """
    Get detailed metadata for a paper.

    Args:
        paper_id: Paper identifier (arXiv ID or Semantic Scholar paper ID)
        source: Data source ("arxiv" or "semantic_scholar")

    Returns:
        Paper metadata dictionary
    """
    if source == "arxiv":
        return await get_arxiv_metadata(paper_id)
    elif source == "semantic_scholar":
        return await get_semantic_scholar_metadata(paper_id)
    else:
        logger.error(f"Unsupported source: {source}")
        return {}


async def get_arxiv_metadata(arxiv_id: str) -> Dict:
    """Get metadata from arXiv."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "id_list": arxiv_id,
        "start": 0,
        "max_results": 1
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(base_url, params=params, timeout=30) as response:
                if response.status == 200:
                    import xml.etree.ElementTree as ET
                    xml_content = await response.text()

                    root = ET.fromstring(xml_content)
                    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                    entry = root.find('atom:entry', namespace)

                    if entry is None:
                        return {}

                    # 提取元数据
                    title_elem = entry.find('atom:title', namespace)
                    title = title_elem.text.strip() if title_elem is not None else ""

                    summary_elem = entry.find('atom:summary', namespace)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""

                    authors = []
                    for author_elem in entry.findall('atom:author', namespace):
                        name_elem = author_elem.find('atom:name', namespace)
                        if name_elem is not None:
                            authors.append(name_elem.text)

                    published_elem = entry.find('atom:published', namespace)
                    published = published_elem.text if published_elem is not None else ""
                    year = int(published[:4]) if published and len(published) >= 4 else 0

                    # 判断是否为综述论文
                    is_survey = any(keyword in title.lower() or keyword in abstract.lower()
                                    for keyword in ['survey', 'review', 'overview'])

                    return {
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "year": year,
                        "venue": "arXiv",
                        "citation_count": None,
                        "doi": None,
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        "source": "arxiv",
                        "is_survey": is_survey,
                        "source_id": arxiv_id
                    }
                else:
                    logger.error(f"arXiv metadata API error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to get arXiv metadata: {e}")
            return {}


async def get_semantic_scholar_metadata(paper_id: str) -> Dict:
    """Get metadata from Semantic Scholar."""
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    params = {
        "fields": "title,authors,year,venue,abstract,citationCount,url,externalIds,openAccessPdf,referenceCount"
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
                    item = await response.json()

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
                                    for keyword in ['survey', 'review', 'overview'])

                    return {
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
                        "source_id": paper_id
                    }
                else:
                    logger.error(f"Semantic Scholar metadata API error: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to get Semantic Scholar metadata: {e}")
            return {}