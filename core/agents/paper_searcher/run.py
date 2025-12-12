from core.agents.paper_searcher import paper_searcher_graph

# 使用Agent进行搜索
async def search_papers():
    query = "小样本学习的最新进展"

    # 初始化状态
    state = {
        "user_query": query,
        "raw_query": query,
        "quality_threshold": 0.7  # 可以自定义质量阈值
    }

    # 运行Agent
    result = await paper_searcher_graph.ainvoke(state)

    # 获取结果
    papers = result.get("final_results", [])
    reasoning = result.get("reasoning", "")
    quality_metrics = result.get("quality_metrics", {})

    print(f"搜索完成，找到 {len(papers)} 篇论文")
    print(f"质量分数: {quality_metrics.get('overall_score', 0):.2f}")
    print(f"搜索解释: {reasoning}")

    # 处理论文数据
    for i, paper in enumerate(papers[:5], 1):
        print(f"{i}. {paper.title}")
        print(f"年份: {paper.year}, 引用: {paper.citation_count or 'N/A'}")
        print(f"来源: {paper.source}, 综述: {paper.is_survey}")
        print()


# 运行
import asyncio

asyncio.run(search_papers())