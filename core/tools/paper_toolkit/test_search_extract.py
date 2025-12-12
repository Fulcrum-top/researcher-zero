# 示例：串联使用Search和Citation工具
async def search_and_analyze_citations(query: str, top_k: int = 10):
    """搜索论文并分析其引用关系"""

    # 1. 使用Search工具查找相关论文
    from core.tools.web_search.tavily_search import tavily_search

    search_results = await tavily_search.ainvoke({
        "queries": [f"{query} survey paper"],
        "max_results": 10
    })

    # 2. 提取论文标题
    paper_titles = []
    for result in search_results[0].get("results", []):
        if "title" in result:
            paper_titles.append(result["title"])

    # 3. 分析这些论文的引用关系
    citation_tool = create_citation_extractor_tool()
    citation_analysis = await citation_tool.extract_and_identify_core_citations(
        paper_identifiers=paper_titles[:5],  # 分析前5篇
        top_k=top_k
    )

    return {
        "search_results": search_results,
        "citation_analysis": citation_analysis
    }