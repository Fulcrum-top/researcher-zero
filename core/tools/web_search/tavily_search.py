import os
import asyncio
from typing import List
from langchain_core.tools import tool
from tavily import AsyncTavilyClient


TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: int = 5):
    tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=True,
            topic="general"
        )
        for query in queries
    ]

    search_results = await asyncio.gather(*search_tasks)
    return search_results
