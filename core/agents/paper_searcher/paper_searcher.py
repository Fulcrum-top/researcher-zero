import asyncio
import json
import hashlib
from datetime import datetime
from typing import Literal, List, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from core.agents.paper_searcher.prompts import (
    get_system_prompt,
    get_query_analysis_prompt,
    get_search_strategy_prompt,
    get_refinement_prompt,
    get_quality_assessment_prompt,
    get_report_generation_prompt
)
from core.agents.paper_searcher.state import (
    PaperSearcherState,
    PaperMetadata,
    SearchPlan,
    SearchStrategy,
    QualityMetrics,
    ConductArxivSearch,
    ConductSemanticScholarSearch,
    GetPaperMetadata,
    GetCitationList,
    DownloadPaperPDF,
    SearchComplete
)
from core.utils.logs import logger
from core.agents.paper_searcher.configuration import PaperSearcherConfig

# 导入工具
from core.tools.paper_toolkit.arxiv_search import arxiv_search
from core.tools.paper_toolkit.semantic_scholar_search import semantic_scholar_search
from core.tools.paper_toolkit.paper_metadata_fetch import get_paper_metadata
from core.tools.paper_toolkit.citation_fetch import get_citation_list
from core.tools.paper_toolkit.pdf_download import download_paper_pdf

# 初始化可配置模型
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "temperature", "model_provider"),
)
logger.info("Configurable model initialized for paper searcher.")


# ==================== 节点函数实现 ====================

async def analyse_query(state: PaperSearcherState, config: RunnableConfig) -> Command[Literal["plan_search"]]:
    """查询分析与意图识别节点"""
    configurable = PaperSearcherConfig.from_runnable_config(config)
    logger.info(f"Analyzing query: {state.user_query}")

    # 准备LLM调用
    model_config = {
        "model": configurable.query_analysis_model,
        "model_provider": "openai",
        "max_tokens": 1000,
        "temperature": 0.1,
        "timeout": 60
    }

    query_analysis_model = configurable_model.with_structured_output(SearchPlan).with_config(model_config)
    query_analysis_prompt = get_query_analysis_prompt(state.user_query)

    try:
        response = await query_analysis_model.ainvoke([HumanMessage(content=query_analysis_prompt)])

        # 记录搜索历史
        state.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "query_analysis",
            "details": {
                "query": state.user_query,
                "parsed_plan": response.dict()
            }
        })

        return Command(
            goto="plan_search",
            update={"search_plan": response}
        )

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        # 创建备用计划
        backup_plan = create_backup_search_plan(state.user_query)
        state.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "query_analysis_fallback",
            "details": {"error": str(e), "backup_plan": backup_plan}
        })

        return Command(
            goto="plan_search",
            update={"search_plan": backup_plan}
        )


def create_backup_search_plan(query: str) -> SearchPlan:
    """创建备用搜索计划"""
    # 简单的关键词提取
    words = query.lower().split()
    keywords = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with', 'about']]

    # 判断是否为寻求综述
    survey_indicators = ['survey', 'review', 'overview', 'introduction', 'state of the art']
    is_survey_query = any(indicator in query.lower() for indicator in survey_indicators)

    return SearchPlan(
        primary_goal="survey" if is_survey_query else "empirical",
        keywords=keywords[:5],
        time_filter="last_3_years",
        expected_count=20
    )


async def plan_search(state: PaperSearcherState, config: RunnableConfig) -> Command[Literal["execute_search"]]:
    """搜索策略规划节点"""
    configurable = PaperSearcherConfig.from_runnable_config(config)
    logger.info(f"Planning search strategy, iteration: {state.search_iteration}")

    # 准备LLM调用
    model_config = {
        "model": configurable.strategy_planning_model,
        "model_provider": "openai",
        "max_tokens": 1000,
        "temperature": 0.1,
        "timeout": 60
    }

    strategy_planning_model = configurable_model.with_structured_output(SearchStrategy).with_config(model_config)

    # 如果是重新搜索，基于历史调整策略
    if state.search_iteration > 0 and state.quality_metrics:
        prompt = get_refinement_prompt(
            state.search_plan.dict() if state.search_plan else {},
            state.quality_metrics.dict() if state.quality_metrics else {},
            state.search_iteration
        )
    else:
        # 初次搜索
        prompt = get_search_strategy_prompt(
            state.search_plan.dict() if state.search_plan else {},
            state.search_iteration
        )

    try:
        response = await strategy_planning_model.ainvoke([HumanMessage(content=prompt)])

        # 记录策略历史
        state.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "plan_search",
            "iteration": state.search_iteration,
            "strategy": response.dict()
        })

        return Command(
            goto="execute_search",
            update={"search_strategy": response}
        )

    except Exception as e:
        logger.error(f"Strategy planning failed: {e}")
        # 创建默认策略
        default_strategy = create_default_strategy(state.search_plan, configurable)

        state.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "strategy_planning_fallback",
            "details": {"error": str(e), "default_strategy": default_strategy.dict()}
        })

        return Command(
            goto="execute_search",
            update={"search_strategy": default_strategy}
        )


def create_default_strategy(search_plan: SearchPlan, config: PaperSearcherConfig) -> SearchStrategy:
    """创建默认搜索策略"""
    sources = []
    query_params = {}

    if config.enable_arxiv and search_plan:
        sources.append("arxiv")
        query_params["arxiv"] = {
            "query": " ".join(search_plan.keywords),
            "max_results": config.max_search_results
        }

    if config.enable_semantic_scholar and search_plan:
        sources.append("semantic_scholar")
        query_params["semantic_scholar"] = {
            "query": " ".join(search_plan.keywords),
            "max_results": config.max_search_results,
            "year_filter": search_plan.time_filter
        }

    return SearchStrategy(
        sources=sources,
        query_params=query_params,
        max_results_per_source=config.max_search_results,
        prioritize_surveys=search_plan.primary_goal == "survey" if search_plan else True,
        iteration=0
    )


async def execute_search(state: PaperSearcherState, config: RunnableConfig) -> Command[Literal["merge_results"]]:
    """执行搜索节点"""
    configurable = PaperSearcherConfig.from_runnable_config(config)
    logger.info(f"Executing search with strategy: {state.search_strategy}")

    if not state.search_strategy:
        logger.warning("No search strategy found, skipping to merge_results")
        return Command(goto="merge_results")

    raw_results = {}
    tools_executed = []

    # 顺序执行每个数据源的搜索
    strategy = state.search_strategy

    for source in strategy.sources:
        try:
            if source == "arxiv" and configurable.enable_arxiv:
                logger.info(f"Executing arXiv search")
                arxiv_params = strategy.query_params.get("arxiv", {})

                # 调用arXiv搜索工具
                arxiv_results = await arxiv_search.ainvoke({
                    "query": arxiv_params.get("query", ""),
                    "max_results": arxiv_params.get("max_results", configurable.max_search_results),
                    "category": arxiv_params.get("category")
                })

                raw_results["arxiv"] = arxiv_results
                tools_executed.append("arxiv_search")

            elif source == "semantic_scholar" and configurable.enable_semantic_scholar:
                logger.info(f"Executing Semantic Scholar search")
                ss_params = strategy.query_params.get("semantic_scholar", {})

                # 调用Semantic Scholar搜索工具
                ss_results = await semantic_scholar_search.ainvoke({
                    "query": ss_params.get("query", ""),
                    "max_results": ss_params.get("max_results", configurable.max_search_results),
                    "year_filter": ss_params.get("year_filter")
                })

                raw_results["semantic_scholar"] = ss_results
                tools_executed.append("semantic_scholar_search")

        except Exception as e:
            logger.error(f"Search execution failed for {source}: {e}")
            raw_results[source] = []

    # 记录执行历史
    state.search_history.append({
        "timestamp": datetime.now().isoformat(),
        "action": "execute_search",
        "iteration": state.search_iteration,
        "tools_executed": tools_executed,
        "results_count": {k: len(v) for k, v in raw_results.items()}
    })

    return Command(
        goto="merge_results",
        update={
            "raw_search_results": raw_results,
            "search_iteration": state.search_iteration + 1
        }
    )


async def merge_results(state: PaperSearcherState, config: RunnableConfig) -> Command[Literal["rank_papers"]]:
    """结果融合与去重节点"""
    logger.info("Merging and deduplicating search results")

    all_papers = []

    # 合并所有来源的结果
    for source, papers in state.raw_search_results.items():
        for paper_dict in papers:
            # 转换为PaperMetadata对象
            try:
                paper = PaperMetadata(**paper_dict)
                all_papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to convert paper: {e}")
                continue

    # 去重
    deduplicated = deduplicate_papers(all_papers)

    # 获取引用列表（可选）
    papers_with_citations = []
    for paper in deduplicated[:10]:  # 只对前10篇获取引用
        try:
            citations = await get_citation_list.ainvoke({
                "paper_id": paper.source_id,
                "max_citations": 20,
                "source": paper.source
            })

            # 提取引用论文标题
            citation_titles = [c.get("title", "") for c in citations if c.get("title")]
            paper.citations = citation_titles
            papers_with_citations.append(paper)

        except Exception as e:
            logger.warning(f"Failed to get citations for {paper.source_id}: {e}")
            paper.citations = []
            papers_with_citations.append(paper)

    # 记录合并历史
    state.search_history.append({
        "timestamp": datetime.now().isoformat(),
        "action": "merge_results",
        "total_papers": len(all_papers),
        "deduplicated_count": len(deduplicated),
        "with_citations_count": len(papers_with_citations)
    })

    return Command(
        goto="rank_papers",
        update={
            "merged_papers": deduplicated,
            "papers_with_citations": papers_with_citations
        }
    )


def deduplicate_papers(papers: List[PaperMetadata]) -> List[PaperMetadata]:
    """论文去重"""
    seen = set()
    unique_papers = []

    for paper in papers:
        if paper.title and paper.authors:
            # 基于标题和第一作者的哈希
            title_hash = hashlib.md5(paper.title.lower().encode()).hexdigest()
            first_author = paper.authors[0].lower() if paper.authors else ""
            paper_hash = f"{title_hash}_{first_author}"

            if paper_hash not in seen:
                seen.add(paper_hash)
                unique_papers.append(paper)

    logger.info(f"Deduplicated: {len(papers)} -> {len(unique_papers)} papers")
    return unique_papers


async def rank_papers(state: PaperSearcherState, config: RunnableConfig) -> Command[Literal["quality_check"]]:
    """智能排序与过滤节点"""
    logger.info(f"Ranking {len(state.papers_with_citations)} papers")

    if not state.papers_with_citations:
        logger.warning("No papers to rank")
        return Command(goto="quality_check")

    # 计算混合排序分数
    ranked_papers = []
    for paper in state.papers_with_citations:
        score = compute_hybrid_score(paper, state.search_plan)
        paper.relevance_score = score
        ranked_papers.append(paper)

    # 按分数降序排序
    ranked_papers.sort(key=lambda x: x.relevance_score, reverse=True)

    # 根据主要目标调整：确保综述论文在前
    if state.search_plan and state.search_plan.primary_goal == "survey":
        ranked_papers.sort(key=lambda x: (not x.is_survey, -x.relevance_score))

    # 记录排序历史
    state.search_history.append({
        "timestamp": datetime.now().isoformat(),
        "action": "rank_papers",
        "ranked_count": len(ranked_papers),
        "top_score": ranked_papers[0].relevance_score if ranked_papers else 0,
        "survey_count": sum(1 for p in ranked_papers if p.is_survey)
    })

    return Command(
        goto="quality_check",
        update={"ranked_papers": ranked_papers}
    )


def compute_hybrid_score(paper: PaperMetadata, search_plan: SearchPlan) -> float:
    """计算混合排序分数"""
    # 权重配置
    weights = {
        "relevance": 0.4,
        "citation": 0.3,
        "recency": 0.2,
        "survey_boost": 0.5
    }

    # 1. 相关性分数
    relevance_score = compute_relevance_score(paper, search_plan.keywords if search_plan else [])

    # 2. 引用分数
    citation_score = compute_citation_score(paper)

    # 3. 时效性分数
    timeliness_score = compute_timeliness_score(paper, search_plan.time_filter if search_plan else None)

    base_score = (
            weights["relevance"] * relevance_score +
            weights["citation"] * citation_score +
            weights["recency"] * timeliness_score
    )

    # 4. 综述额外加分
    if paper.is_survey and search_plan and search_plan.primary_goal == "survey":
        base_score += weights["survey_boost"]

    return min(base_score, 1.0)


def compute_relevance_score(paper: PaperMetadata, keywords: List[str]) -> float:
    """计算相关性分数"""
    if not keywords:
        return 0.5

    text = f"{paper.title} {paper.abstract}".lower()
    score = 0.0

    for keyword in keywords:
        if keyword.lower() in text:
            # 根据出现位置和频率加分
            keyword_lower = keyword.lower()
            title_score = paper.title.lower().count(keyword_lower) * 2
            abstract_score = paper.abstract.lower().count(keyword_lower) * 1
            score += (title_score + abstract_score)

    return min(score / 10.0, 1.0)


def compute_citation_score(paper: PaperMetadata) -> float:
    """计算引用分数"""
    if paper.citation_count is None:
        return 0.5

    if paper.citation_count == 0:
        return 0.1
    else:
        import math
        return min(math.log10(paper.citation_count + 1) / 3.0, 1.0)


def compute_timeliness_score(paper: PaperMetadata, time_filter: str = None) -> float:
    """计算时效性分数"""
    current_year = datetime.now().year

    if paper.year == 0:
        return 0.5

    age = current_year - paper.year

    if time_filter == "last_1_year":
        if age <= 1:
            return 1.0
        else:
            return max(0, 1.0 - (age - 1) * 0.3)
    elif time_filter == "last_3_years":
        if age <= 3:
            return 1.0 - (age * 0.1)
        else:
            return max(0, 0.7 - (age - 3) * 0.2)
    elif time_filter == "last_5_years":
        if age <= 5:
            return 1.0 - (age * 0.1)
        else:
            return max(0, 0.5 - (age - 5) * 0.05)
    else:
        # 默认：近5年高分，之后逐渐降低
        if age <= 5:
            return 1.0 - (age * 0.1)
        else:
            return max(0, 0.5 - (age - 5) * 0.05)


async def quality_check(state: PaperSearcherState, config: RunnableConfig) -> Command[
    Literal["plan_search", "generate_report"]]:
    """结果质量评估节点"""
    configurable = PaperSearcherConfig.from_runnable_config(config)
    logger.info("Performing quality assessment")

    if not state.ranked_papers:
        logger.warning("No papers to assess quality")
        # 如果没有论文，返回低质量分，触发重新搜索
        quality_metrics = QualityMetrics(
            breadth_score=0.0,
            depth_score=0.0,
            timeliness_score=0.0,
            overall_score=0.0,
            issues=["No papers found"],
            suggestions=["Try different keywords or expand search scope"]
        )

        return Command(
            goto="plan_search",
            update={"quality_metrics": quality_metrics}
        )

    # 准备LLM调用进行质量评估
    model_config = {
        "model": configurable.quality_assessment_model,
        "model_provider": "openai",
        "max_tokens": 1000,
        "temperature": 0.1,
        "timeout": 60
    }

    quality_model = configurable_model.with_structured_output(QualityMetrics).with_config(model_config)

    # 准备评估数据
    papers_data = [p.dict() for p in state.ranked_papers[:20]]  # 只评估前20篇
    strategy_data = state.search_strategy.dict() if state.search_strategy else {}

    prompt = get_quality_assessment_prompt(papers_data, strategy_data)

    try:
        quality_metrics = await quality_model.ainvoke([HumanMessage(content=prompt)])

    except Exception as e:
        logger.error(f"Quality assessment failed, using fallback: {e}")
        # 使用备用方法计算质量分
        quality_metrics = compute_fallback_quality_metrics(state.ranked_papers, state.search_plan)

    # 记录质量评估历史
    state.search_history.append({
        "timestamp": datetime.now().isoformat(),
        "action": "quality_check",
        "iteration": state.search_iteration,
        "metrics": quality_metrics.dict(),
        "threshold": state.quality_threshold,
        "passed": quality_metrics.overall_score >= state.quality_threshold
    })

    # 检查是否需要重新搜索
    if quality_metrics.overall_score < state.quality_threshold:
        # 检查是否超过最大迭代次数
        if state.search_iteration < configurable.max_search_iterations:
            logger.info(
                f"Quality insufficient ({quality_metrics.overall_score:.2f} < {state.quality_threshold:.2f}), refining strategy")
            return Command(
                goto="plan_search",
                update={"quality_metrics": quality_metrics}
            )
        else:
            logger.info(f"Max iterations reached ({state.search_iteration}), proceeding with current results")

    logger.info(
        f"Quality sufficient ({quality_metrics.overall_score:.2f} >= {state.quality_threshold:.2f}), generating report")
    return Command(
        goto="generate_report",
        update={"quality_metrics": quality_metrics}
    )


def compute_fallback_quality_metrics(papers: List[PaperMetadata], search_plan: SearchPlan) -> QualityMetrics:
    """备用质量评估方法"""
    if not papers:
        return QualityMetrics(
            breadth_score=0.0,
            depth_score=0.0,
            timeliness_score=0.0,
            overall_score=0.0,
            issues=["No papers found"],
            suggestions=["Try different keywords"]
        )

    # 计算广度分数
    breadth_score = compute_breadth_score(papers, search_plan)

    # 计算深度分数
    depth_score = compute_depth_score(papers)

    # 计算时间有效性分数
    timeliness_score = compute_timeliness_compliance(papers, search_plan)

    # 计算总体分数
    overall_score = 0.3 * breadth_score + 0.4 * depth_score + 0.3 * timeliness_score

    issues = []
    suggestions = []

    if overall_score < 0.7:
        issues.append("Overall quality is below expectation")
        suggestions.append("Consider expanding search scope or adjusting keywords")

    if len(papers) < 5:
        issues.append("Insufficient number of papers")
        suggestions.append("Increase max_results or broaden search terms")

    return QualityMetrics(
        breadth_score=breadth_score,
        depth_score=depth_score,
        timeliness_score=timeliness_score,
        overall_score=overall_score,
        issues=issues,
        suggestions=suggestions
    )


def compute_breadth_score(papers: List[PaperMetadata], search_plan: SearchPlan) -> float:
    """计算广度分数"""
    if not search_plan or not search_plan.keywords:
        return 0.5

    keywords = search_plan.keywords
    coverage = 0.0

    for keyword in keywords:
        keyword_found = False
        for paper in papers[:10]:  # 只检查前10篇
            if keyword.lower() in paper.title.lower() or keyword.lower() in paper.abstract.lower():
                keyword_found = True
                break

        if keyword_found:
            coverage += 1

    return coverage / len(keywords) if keywords else 0.5


def compute_depth_score(papers: List[PaperMetadata]) -> float:
    """计算深度分数"""
    if not papers:
        return 0.0

    scores = []

    for paper in papers[:10]:  # 只评估前10篇
        paper_score = 0.0

        # 1. 引用数得分
        if paper.citation_count:
            if paper.citation_count > 100:
                paper_score += 0.4
            elif paper.citation_count > 10:
                paper_score += 0.2
            else:
                paper_score += 0.1

        # 2. 来源质量得分
        if paper.source == "semantic_scholar":
            paper_score += 0.3
        elif paper.source == "arxiv":
            paper_score += 0.2

        # 3. 综述论文额外得分
        if paper.is_survey:
            paper_score += 0.3

        scores.append(min(paper_score, 1.0))

    return sum(scores) / len(scores) if scores else 0.5


def compute_timeliness_compliance(papers: List[PaperMetadata], search_plan: SearchPlan) -> float:
    """计算时间有效性分数"""
    if not papers:
        return 0.0

    current_year = datetime.now().year
    time_filter = search_plan.time_filter if search_plan else None

    if time_filter == "last_1_year":
        recent_count = sum(1 for p in papers if current_year - p.year <= 1)
        return recent_count / len(papers)
    elif time_filter == "last_3_years":
        recent_count = sum(1 for p in papers if current_year - p.year <= 3)
        return recent_count / len(papers)
    elif time_filter == "last_5_years":
        recent_count = sum(1 for p in papers if current_year - p.year <= 5)
        return recent_count / len(papers)
    else:
        # 无时间限制，默认给中等分数
        return 0.7


async def generate_report(state: PaperSearcherState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """生成报告节点"""
    configurable = PaperSearcherConfig.from_runnable_config(config)
    logger.info("Generating final report")

    # 只取Top 20篇论文
    top_papers = state.ranked_papers[:20]

    # 下载PDF（如果配置允许）
    if configurable.enable_pdf_download:
        await download_pdfs_for_top_papers(top_papers)

    # 生成搜索过程解释
    model_config = {
        "model": configurable.report_generation_model,
        "model_provider": "openai",
        "max_tokens": 1000,
        "temperature": 0.3,
        "timeout": 60
    }

    report_model = configurable_model.with_config(model_config)

    prompt = get_report_generation_prompt(
        state.user_query,
        state.search_plan.dict() if state.search_plan else {},
        state.search_strategy.dict() if state.search_strategy else {},
        [p.dict() for p in top_papers],
        state.quality_metrics.dict() if state.quality_metrics else {}
    )

    try:
        reasoning = await report_model.ainvoke([HumanMessage(content=prompt)])

        # 构建最终输出
        final_results = top_papers

        # 记录报告生成历史
        state.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "generate_report",
            "final_papers_count": len(final_results),
            "has_pdfs": configurable.enable_pdf_download
        })

        return Command(
            goto=END,
            update={
                "final_results": final_results,
                "reasoning": reasoning.content if hasattr(reasoning, 'content') else str(reasoning),
                "final_report": reasoning.content if hasattr(reasoning, 'content') else str(reasoning)
            }
        )

    except Exception as e:
        logger.error(f"Report generation failed: {e}")

        # 使用备用报告
        backup_report = create_backup_report(state.user_query, top_papers)

        return Command(
            goto=END,
            update={
                "final_results": top_papers,
                "reasoning": backup_report,
                "final_report": backup_report
            }
        )


async def download_pdfs_for_top_papers(papers: List[PaperMetadata]):
    """为Top论文下载PDF"""
    logger.info(f"Downloading PDFs for top {len(papers)} papers")

    download_tasks = []
    for paper in papers[:5]:  # 只下载前5篇的PDF
        if paper.pdf_url and paper.source in ["arxiv", "semantic_scholar"]:
            task = download_paper_pdf.ainvoke({
                "paper_id": paper.source_id,
                "source": paper.source
            })
            download_tasks.append(task)

    if download_tasks:
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        logger.info(f"PDF download completed: {successful}/{len(download_tasks)} successful")


def create_backup_report(query: str, papers: List[PaperMetadata]) -> str:
    """创建备用报告"""
    if not papers:
        return f"搜索查询 '{query}' 未找到相关论文。请尝试调整搜索词或放宽搜索条件。"

    paper_count = len(papers)
    survey_count = sum(1 for p in papers if p.is_survey)
    recent_count = sum(1 for p in papers if datetime.now().year - p.year <= 3)

    return f"""基于查询 '{query}' 的搜索结果：

本次搜索共找到 {paper_count} 篇相关论文，其中：
- 综述论文：{survey_count} 篇
- 近3年论文：{recent_count} 篇
- 主要来源：{', '.join(set(p.source for p in papers))}

代表性论文：
{chr(10).join([f'{i + 1}. {p.title[:60]}...' for i, p in enumerate(papers[:3])])}

建议：可根据需要进一步调整搜索策略或下载全文进行深度阅读。"""


# ==================== 构建状态机 ====================

paper_searcher_graph = StateGraph(PaperSearcherState)

# 添加节点
paper_searcher_graph.add_node("analyse_query", analyse_query)
paper_searcher_graph.add_node("plan_search", plan_search)
paper_searcher_graph.add_node("execute_search", execute_search)
paper_searcher_graph.add_node("merge_results", merge_results)
paper_searcher_graph.add_node("rank_papers", rank_papers)
paper_searcher_graph.add_node("quality_check", quality_check)
paper_searcher_graph.add_node("generate_report", generate_report)

# 设置边
paper_searcher_graph.add_edge(START, "analyse_query")
paper_searcher_graph.add_edge("analyse_query", "plan_search")
paper_searcher_graph.add_edge("plan_search", "execute_search")
paper_searcher_graph.add_edge("execute_search", "merge_results")
paper_searcher_graph.add_edge("merge_results", "rank_papers")
paper_searcher_graph.add_edge("rank_papers", "quality_check")

# 条件边：质量检查后决定下一步
paper_searcher_graph.add_conditional_edges(
    "quality_check",
    lambda state: "plan_search" if (
            state.quality_metrics and
            state.quality_metrics.overall_score < state.quality_threshold and
            state.search_iteration < 3  # 最大3次迭代
    ) else "generate_report"
)

# 固定边
paper_searcher_graph.add_edge("generate_report", END)

# 编译状态机
paper_searcher_graph = paper_searcher_graph.compile()


# ==================== 主函数和测试 ====================

async def main():
    """测试函数"""
    sample_query = "强化学习中的探索与利用平衡最新研究"

    state = PaperSearcherState(
        user_query=sample_query,
        raw_query=sample_query,
        quality_threshold=0.7
    )

    result = await paper_searcher_graph.ainvoke(state)

    logger.info(f"搜索完成，找到 {len(result.get('final_results', []))} 篇论文")
    logger.info(f"质量评估: {result.get('quality_metrics', {}).get('overall_score', 0):.2f}")
    logger.info(f"搜索解释: {result.get('reasoning', '')[:200]}...")

    if result.get('final_results'):
        logger.info("\nTop 3 论文:")
        for i, paper in enumerate(result['final_results'][:3], 1):
            logger.info(f"{i}. {paper.title[:60]}...")
            logger.info(f"   作者: {', '.join(paper.authors[:2])}")
            logger.info(f"   年份: {paper.year}, 来源: {paper.source}")
            logger.info(f"   引用: {paper.citation_count or 'N/A'}, 综述: {paper.is_survey}")
            logger.info(f"   分数: {paper.relevance_score:.3f}")
            logger.info("")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())