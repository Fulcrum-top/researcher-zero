# core/tools/paper_toolkit/search_tool.py

import os
import json
import asyncio
import aiohttp
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, TypedDict
from enum import Enum

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.logs import logger

# ==================== 数据模型定义 ====================

class PaperMetadata(TypedDict):
    """论文元数据结构"""
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: str
    citation_count: Optional[int]
    doi: Optional[str]
    url: str
    pdf_url: Optional[str]
    source: str
    is_survey: bool
    source_id: str
    relevance_score: float
    citations: List[str]  # 引用文献列表（存储引用论文的标题），通过工具2获取


class SearchGoal(str, Enum):
    """搜索目标枚举"""
    FIND_SURVEYS = "find_surveys"
    FIND_EMPIRICAL = "find_empirical_studies"
    FIND_LATEST = "find_latest_advances"
    FIND_HIGHLY_CITED = "find_highly_cited"


class SearchPlan(BaseModel):
    """搜索计划"""
    primary_goal: SearchGoal
    secondary_goals: List[SearchGoal] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    time_filter: Optional[str] = None
    must_include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)


# ==================== 数据源适配器 ====================

class BaseSearchAdapter:
    """数据源适配器基类"""

    def __init__(self, name: str, max_results: int = 50):
        self.name = name
        self.max_results = max_results

    def build_query(self, plan: SearchPlan) -> Any:
        """构建查询"""
        raise NotImplementedError

    async def search(self, plan: SearchPlan) -> List[PaperMetadata]:
        """执行搜索"""
        raise NotImplementedError

    def _is_survey(self, title: str, abstract: str) -> bool:
        """判断是否为综述论文"""
        survey_keywords = ['survey', 'review', 'overview', 'state of the art',
                           'comprehensive study', 'literature review']
        text = f"{title.lower()} {abstract.lower()}"
        return any(keyword in text for keyword in survey_keywords)

# ==================== Semantic Scholar 适配器 ====================

class SemanticScholarAdapter(BaseSearchAdapter):
    """Semantic Scholar 适配器"""

    def __init__(self):
        super().__init__("semantic_scholar")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def build_query(self, plan: SearchPlan) -> Dict[str, Any]:
        """构建查询参数"""
        query_params = {
            "query": " ".join(plan.keywords),
            "limit": self.max_results,
            "fields": "title,authors,year,venue,abstract,citationCount,url,externalIds,openAccessPdf",
            "sort": "relevance" if plan.primary_goal == SearchGoal.FIND_SURVEYS else "citationCount:desc"
        }

        if plan.time_filter:
            current_year = datetime.now().year
            if plan.time_filter == "last_1_year":
                query_params["year"] = f"{current_year - 1}-{current_year}"
            elif plan.time_filter == "last_3_years":
                query_params["year"] = f"{current_year - 3}-{current_year}"
            elif plan.time_filter == "last_5_years":
                query_params["year"] = f"{current_year - 5}-{current_year}"

        return query_params

    async def search(self, plan: SearchPlan) -> List[PaperMetadata]:
        """执行搜索"""
        query_params = self.build_query(plan)
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                        f"{self.base_url}/paper/search",
                        params=query_params,
                        headers=headers,
                        timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_results(data.get("data", []))
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Semantic Scholar search failed: {e}")
                return []

    def _parse_results(self, results: List[Dict]) -> List[PaperMetadata]:
        """解析结果"""
        papers = []

        for item in results:
            try:
                paper = PaperMetadata(
                    title=item.get("title", ""),
                    authors=[author.get("name", "") for author in item.get("authors", [])],
                    abstract=item.get("abstract", ""),
                    year=item.get("year", 0),
                    venue=item.get("venue", ""),
                    citation_count=item.get("citationCount", 0),
                    doi=item.get("externalIds", {}).get("DOI"),
                    url=item.get("url", ""),
                    pdf_url=item.get("openAccessPdf", {}).get("url"),
                    source=self.name,
                    is_survey=self._is_survey(
                        item.get("title", ""),
                        item.get("abstract", "")
                    ),
                    source_id=item.get("paperId", ""),
                    relevance_score=0.0,
                    citations=[]  # 搜索阶段先设为空，后续由工具2填充
                )
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse paper: {e}")
                continue

        return papers

# ==================== arXiv 适配器 ====================

class ArxivAdapter(BaseSearchAdapter):
    """arXiv 适配器"""

    def __init__(self):
        super().__init__("arxiv")
        self.base_url = "http://export.arxiv.org/api/query"

    def build_query(self, plan: SearchPlan) -> Dict[str, Any]:
        """构建查询参数"""
        query_parts = []

        if plan.keywords:
            query_parts.append(f"all:{' AND '.join(plan.keywords)}")

        if plan.time_filter:
            if plan.time_filter == "last_1_year":
                query_parts.append("submittedDate:[NOW-365DAYS TO NOW]")
            elif plan.time_filter == "last_3_years":
                query_parts.append("submittedDate:[NOW-1095DAYS TO NOW]")

        for term in plan.must_include:
            query_parts.append(f"abs:{term}")

        query_str = " AND ".join(query_parts) if query_parts else "all:*"

        return {
            "search_query": query_str,
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

    async def search(self, plan: SearchPlan) -> List[PaperMetadata]:
        """执行搜索"""
        query_params = self.build_query(plan)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                        self.base_url,
                        params=query_params,
                        timeout=30
                ) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        return self._parse_results(xml_content)
                    else:
                        logger.error(f"arXiv API error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"arXiv search failed: {e}")
                return []

    def _parse_results(self, xml_content: str) -> List[PaperMetadata]:
        """解析结果"""
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}

            papers = []
            for entry in root.findall('atom:entry', namespace):
                try:
                    title = entry.find('atom:title', namespace).text.strip()
                    summary = entry.find('atom:summary', namespace).text.strip() if entry.find('atom:summary',
                                                                                               namespace) is not None else ""

                    authors = []
                    for author_elem in entry.findall('atom:author', namespace):
                        name_elem = author_elem.find('atom:name', namespace)
                        if name_elem is not None:
                            authors.append(name_elem.text)

                    published = entry.find('atom:published', namespace).text
                    year = int(published[:4]) if published else 0

                    arxiv_id = entry.find('atom:id', namespace).text.split('/')[-1]

                    paper = PaperMetadata(
                        title=title,
                        authors=authors,
                        abstract=summary,
                        year=year,
                        venue="arXiv",
                        citation_count=None,
                        doi=None,
                        url=f"https://arxiv.org/abs/{arxiv_id}",
                        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                        source=self.name,
                        is_survey=self._is_survey(title, summary),
                        source_id=arxiv_id,
                        relevance_score=0.0,
                        citations=[]  # arXiv API不提供引用列表，后续由工具2获取
                    )
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse arXiv entry: {e}")
                    continue

            return papers
        except Exception as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
            return []


# ==================== LLM 客户端 ====================

class LLMClient:
    """LLM客户端，通过llm-gateway调用"""

    def __init__(self):
        self.api_base = os.getenv("LLM_GATEWAY_URL", "http://localhost:4000")
        self.api_key = os.getenv("LLM_GATEWAY_API_KEY", "sk-local-dev")
        self.model = os.getenv("LLM_MODEL", "kimi")  # 默认使用kimi

    async def acomplete(self, prompt: str, **kwargs) -> str:
        """异步调用LLM完成请求"""
        import httpx

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"LLM Gateway error: {e}")
                raise


# ==================== 混合排序算法 ====================

class PaperRanker:
    """论文排序器"""

    def __init__(self):
        self.weights = {
            'relevance': 0.4,
            'citation': 0.3,
            'recency': 0.2,
            'survey_boost': 0.5
        }

    def compute_relevance_score(self, paper: PaperMetadata, keywords: List[str]) -> float:
        """计算相关性分数"""
        text = f"{paper['title']} {paper['abstract']}".lower()
        score = 0.0

        for keyword in keywords:
            if keyword.lower() in text:
                keyword_lower = keyword.lower()
                title_score = paper['title'].lower().count(keyword_lower) * 2
                abstract_score = paper['abstract'].lower().count(keyword_lower) * 1
                score += (title_score + abstract_score)

        return min(score / 10.0, 1.0)

    def compute_recency_score(self, paper: PaperMetadata) -> float:
        """计算时效性分数"""
        current_year = datetime.now().year
        if paper['year'] == 0:
            return 0.5

        age = current_year - paper['year']
        if age <= 1:
            return 1.0
        elif age <= 3:
            return 0.8
        elif age <= 5:
            return 0.5
        else:
            return 0.2

    def compute_citation_score(self, paper: PaperMetadata) -> float:
        """计算引用分数"""
        if paper['citation_count'] is None:
            return 0.5

        if paper['citation_count'] == 0:
            return 0.1
        else:
            import math
            return min(math.log10(paper['citation_count'] + 1) / 3.0, 1.0)

    def compute_hybrid_score(self, paper: PaperMetadata, plan: SearchPlan) -> float:
        """计算混合排序分数"""
        relevance = self.compute_relevance_score(paper, plan.keywords)
        citation = self.compute_citation_score(paper)
        recency = self.compute_recency_score(paper)

        base_score = (
                self.weights['relevance'] * relevance +
                self.weights['citation'] * citation +
                self.weights['recency'] * recency
        )

        if paper['is_survey'] and SearchGoal.FIND_SURVEYS in [plan.primary_goal] + plan.secondary_goals:
            base_score += self.weights['survey_boost']

        return min(base_score, 1.0)


# ==================== 搜索工具 ====================

SEARCH_TOOL_DESCRIPTION = (
    "智能搜索学术论文，特别擅长定位领域内的综述论文和高影响力研究。"
    "输入应为自然语言描述的研究领域（如'强化学习的最新进展'或'小样本学习综述'）。"
    "工具会自动分析查询意图，并行搜索多个学术数据库，并智能排序结果。"
)


@tool(description=SEARCH_TOOL_DESCRIPTION)
async def research_paper_search(
        query: str,
        max_results: int = 20
) -> Dict[str, Any]:
    """
    智能搜索学术论文。

    Args:
        query: 对研究领域的自然语言描述
        max_results: 期望返回的最大论文数量，默认20

    Returns:
        包含搜索结果和元信息的字典
    """
    # 初始化组件
    llm_client = LLMClient()
    ranker = PaperRanker()

    # 1. 查询分析与规划
    logger.info(f"Analyzing query: {query}")
    search_plan = await _plan_search(query, llm_client)

    # 2. 多源并行检索
    logger.info(f"Searching with plan: {search_plan.dict()}")
    all_papers = await _search_all_sources(search_plan)

    # 3. 去重
    deduplicated_papers = _deduplicate_papers(all_papers)

    # 4. 排序
    for paper in deduplicated_papers:
        paper['relevance_score'] = ranker.compute_hybrid_score(paper, search_plan)

    # 根据主要目标调整排序
    if search_plan.primary_goal == SearchGoal.FIND_SURVEYS:
        deduplicated_papers.sort(key=lambda x: (not x['is_survey'], -x['relevance_score']))
    else:
        deduplicated_papers.sort(key=lambda x: -x['relevance_score'])

    ranked_papers = deduplicated_papers[:max_results]

    # 5. 生成解释
    reasoning = await _generate_reasoning(query, search_plan, ranked_papers, llm_client)

    # 6. 统计源贡献
    source_stats = _calculate_source_stats(ranked_papers)

    # 7. 返回结果
    return {
        "papers": ranked_papers,
        "search_strategy": search_plan.dict(),
        "reasoning": reasoning,
        "source_stats": source_stats,
        "query_time": datetime.now().isoformat(),
        "query": query
    }


async def _plan_search(query: str, llm_client: LLMClient) -> SearchPlan:
    """分析查询并生成搜索计划"""
    planner_prompt = f"""
    你是一个专业的学术研究助理。请分析以下用户查询，并制定一个精准的文献搜索计划。

    查询：{query}

    请以JSON格式输出，包含以下字段：
    1. primary_goal: 主要目标，可选值：find_surveys, find_empirical_studies, find_latest_advances, find_highly_cited
    2. secondary_goals: 次要目标列表，可选值同上
    3. keywords: 关键词列表，用于搜索
    4. time_filter: 时间过滤，可选值：last_1_year, last_3_years, last_5_years 或 null
    5. must_include: 必须包含的术语列表
    6. exclude: 排除的术语列表

    思考过程：
    1. 判断查询意图：是寻求领域概览、特定方法、最新进展还是经典论文？
    2. 提取核心学术概念、方法、技术名词
    3. 判断时效性要求
    4. 确定是否需要特别强调综述论文

    示例输出：
    {{
        "primary_goal": "find_surveys",
        "secondary_goals": ["find_highly_cited"],
        "keywords": ["reinforcement learning", "exploration", "exploitation", "balance"],
        "time_filter": "last_3_years",
        "must_include": ["survey", "review"],
        "exclude": ["biology", "chemistry"]
    }}
    """

    try:
        response = await llm_client.acomplete(
            prompt=planner_prompt,
            temperature=0.1,
            max_tokens=500
        )

        # 提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            plan_dict = json.loads(json_match.group())
            return SearchPlan(**plan_dict)
        else:
            logger.warning("Failed to parse LLM response as JSON, using fallback")
            return _create_fallback_plan(query)

    except Exception as e:
        logger.error(f"Search planning failed: {e}")
        return _create_fallback_plan(query)


def _create_fallback_plan(query: str) -> SearchPlan:
    """创建备用搜索计划"""
    words = query.lower().split()
    keywords = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with', 'about']]

    survey_indicators = ['survey', 'review', 'overview', 'introduction', 'state of the art']
    is_survey_query = any(indicator in query.lower() for indicator in survey_indicators)

    return SearchPlan(
        primary_goal=SearchGoal.FIND_SURVEYS if is_survey_query else SearchGoal.FIND_EMPIRICAL,
        keywords=keywords[:5],
        time_filter="last_3_years"
    )


async def _search_all_sources(plan: SearchPlan) -> List[PaperMetadata]:
    """并发搜索所有数据源"""
    # 初始化适配器
    adapters = []

    # 根据配置决定启用哪些数据源
    enable_semantic_scholar = os.getenv("ENABLE_SEMANTIC_SCHOLAR", "true").lower() == "true"
    enable_arxiv = os.getenv("ENABLE_ARXIV", "true").lower() == "true"

    if enable_semantic_scholar:
        adapters.append(SemanticScholarAdapter())

    if enable_arxiv:
        adapters.append(ArxivAdapter())

    if not adapters:
        logger.warning("No search sources enabled!")
        return []

    # 并发搜索
    tasks = [adapter.search(plan) for adapter in adapters]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 合并结果
    all_papers = []
    for i, result in enumerate(results):
        adapter_name = adapters[i].name if i < len(adapters) else "unknown"
        if isinstance(result, Exception):
            logger.error(f"Source {adapter_name} failed: {result}")
            continue
        all_papers.extend(result)

    logger.info(f"Retrieved {len(all_papers)} papers from {len(adapters)} sources")
    return all_papers


def _deduplicate_papers(papers: List[PaperMetadata]) -> List[PaperMetadata]:
    """论文去重"""
    seen = set()
    unique_papers = []

    for paper in papers:
        if paper['title'] and paper['authors']:
            title_hash = hashlib.md5(paper['title'].lower().encode()).hexdigest()
            first_author = paper['authors'][0].lower() if paper['authors'] else ""
            paper_hash = f"{title_hash}_{first_author}"

            if paper_hash not in seen:
                seen.add(paper_hash)
                unique_papers.append(paper)

    logger.info(f"Deduplicated: {len(papers)} -> {len(unique_papers)}")
    return unique_papers


async def _generate_reasoning(
        query: str,
        plan: SearchPlan,
        papers: List[PaperMetadata],
        llm_client: LLMClient
) -> str:
    """生成搜索解释"""
    if not papers:
        return "未找到相关论文。请尝试调整查询词或放宽搜索条件。"

    top_titles = [p['title'] for p in papers[:3]]

    reasoning_prompt = f"""
    基于以下搜索信息，为用户生成一段简洁、专业的搜索过程解释：

    原始查询：{query}
    搜索策略：{plan.dict()}
    返回论文数量：{len(papers)}篇
    代表性论文：
    {chr(10).join([f'- {title}' for title in top_titles])}

    请生成一段2-3句话的解释，说明：
    1. 搜索的重点（如是否侧重综述、时效性、高影响力等）
    2. 返回结果的主要特点
    3. 任何需要注意的事项（如某些领域覆盖有限）

    使用中文回答，保持专业但友好的语气。
    """

    try:
        response = await llm_client.acomplete(
            prompt=reasoning_prompt,
            temperature=0.3,
            max_tokens=300
        )
        return response.strip()
    except Exception as e:
        logger.error(f"Failed to generate reasoning: {e}")
        return "搜索已完成。"


def _calculate_source_stats(papers: List[PaperMetadata]) -> Dict[str, int]:
    """计算源贡献统计"""
    stats = {}
    for paper in papers:
        source = paper['source']
        stats[source] = stats.get(source, 0) + 1
    return stats


# ==================== 工具工厂函数 ====================

def get_paper_search_tool():
    """获取论文搜索工具实例"""
    return research_paper_search


# ==================== 测试代码 ====================

async def test_search():
    """测试搜索工具"""
    # 设置测试环境变量
    os.environ["LLM_GATEWAY_URL"] = "http://localhost:4000"
    os.environ["LLM_GATEWAY_API_KEY"] = "sk-local-dev"
    os.environ["LLM_MODEL"] = "kimi"

    print("🔍 测试学术论文搜索工具...")

    # 模拟LLM响应
    async def mock_plan_search(query, llm_client):
        return SearchPlan(
            primary_goal=SearchGoal.FIND_SURVEYS,
            keywords=["reinforcement learning", "deep learning"],
            time_filter="last_3_years"
        )

    # 临时替换函数进行测试
    original_plan_search = _plan_search
    import core.tools.paper_toolkit.search_tool as module
    module._plan_search = mock_plan_search

    try:
        result = await research_paper_search(
            query="强化学习的最新综述",
            max_results=5
        )

        print(f"📊 找到论文数量: {len(result['papers'])}")
        print(f"🎯 搜索策略: {result['search_strategy']}")
        print(f"💡 解释说明: {result['reasoning']}")

        if result['papers']:
            print("\n📄 前3篇论文:")
            for i, paper in enumerate(result['papers'][:3], 1):
                print(f"{i}. {paper['title'][:80]}...")
                print(f"   作者: {', '.join(paper['authors'][:2])}")
                print(f"   年份: {paper['year']}, 来源: {paper['source']}")
                print(f"   综述: {paper['is_survey']}, 分数: {paper['relevance_score']:.3f}")
                print(f"   引用数: {len(paper['citations'])}")
                print()
    finally:
        # 恢复原函数
        module._plan_search = original_plan_search


if __name__ == "__main__":
    # 运行测试
    import asyncio
    asyncio.run(test_search())