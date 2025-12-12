import os
import asyncio
import aiohttp
from typing import Dict
from langchain_core.tools import tool
from core.utils.logs import logger

PDF_DOWNLOAD_DESCRIPTION = (
    "Download PDF for a paper from arXiv or Semantic Scholar. "
    "Saves PDF to local storage and returns download information."
)


@tool(description=PDF_DOWNLOAD_DESCRIPTION)
async def download_paper_pdf(
        paper_id: str,
        source: str = "arxiv"
) -> Dict:
    """
    Download PDF for a paper.

    Args:
        paper_id: Paper identifier
        source: Data source ("arxiv" or "semantic_scholar")

    Returns:
        Download information dictionary
    """
    if source == "arxiv":
        return await download_arxiv_pdf(paper_id)
    elif source == "semantic_scholar":
        return await download_semantic_scholar_pdf(paper_id)
    else:
        logger.error(f"Unsupported source for PDF download: {source}")
        return {"success": False, "error": f"Unsupported source: {source}"}


async def download_arxiv_pdf(arxiv_id: str) -> Dict:
    """Download PDF from arXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    save_dir = os.getenv("PAPER_DOWNLOAD_DIR", "./data/papers")

    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{arxiv_id}.pdf")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(pdf_url, timeout=60) as response:
                if response.status == 200:
                    pdf_content = await response.read()

                    with open(save_path, 'wb') as f:
                        f.write(pdf_content)

                    logger.info(f"Downloaded arXiv PDF: {arxiv_id} -> {save_path}")

                    return {
                        "success": True,
                        "paper_id": arxiv_id,
                        "source": "arxiv",
                        "pdf_url": pdf_url,
                        "local_path": save_path,
                        "file_size": len(pdf_content)
                    }
                else:
                    logger.error(f"arXiv PDF download error: {response.status}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "paper_id": arxiv_id
                    }
        except Exception as e:
            logger.error(f"Failed to download arXiv PDF: {e}")
            return {
                "success": False,
                "error": str(e),
                "paper_id": arxiv_id
            }


async def download_semantic_scholar_pdf(paper_id: str) -> Dict:
    """Download PDF from Semantic Scholar if available."""
    # 首先获取论文元数据以获取PDF URL
    from .paper_metadata_fetch import get_semantic_scholar_metadata
    metadata = await get_semantic_scholar_metadata(paper_id)

    pdf_url = metadata.get("pdf_url")
    if not pdf_url:
        logger.warning(f"No PDF URL available for paper {paper_id}")
        return {
            "success": False,
            "error": "No PDF URL available",
            "paper_id": paper_id
        }

    save_dir = os.getenv("PAPER_DOWNLOAD_DIR", "./data/papers")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{paper_id}.pdf")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(pdf_url, timeout=60) as response:
                if response.status == 200:
                    pdf_content = await response.read()

                    with open(save_path, 'wb') as f:
                        f.write(pdf_content)

                    logger.info(f"Downloaded Semantic Scholar PDF: {paper_id} -> {save_path}")

                    return {
                        "success": True,
                        "paper_id": paper_id,
                        "source": "semantic_scholar",
                        "pdf_url": pdf_url,
                        "local_path": save_path,
                        "file_size": len(pdf_content)
                    }
                else:
                    logger.error(f"Semantic Scholar PDF download error: {response.status}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "paper_id": paper_id
                    }
        except Exception as e:
            logger.error(f"Failed to download Semantic Scholar PDF: {e}")
            return {
                "success": False,
                "error": str(e),
                "paper_id": paper_id
            }