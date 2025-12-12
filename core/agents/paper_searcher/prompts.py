def get_system_prompt() -> str:
    return """You are a professional academic research assistant specialized in literature search.
Your goal is to help researchers find the most relevant and high-quality academic papers.
"""


def get_query_analysis_prompt(query: str) -> str:
    return f"""
You are a professional academic research assistant. Please analyze the following user query and develop a precise literature search plan.

User Query: {query}

Please analyze the following aspects:
1. **Query Type**: Is the user seeking an overview of the field (needs Survey) or in-depth research on a specific topic (needs core literature)?
2. **Core Concepts**: Extract key academic terms, method names, technical terms
3. **Time Scope**: Does the user imply recency requirements?
4. **Quality Requirements**: Does the user need high-impact papers?

Output in JSON format with the following fields:
{{
    "primary_goal": "survey" or "empirical",
    "keywords": ["keyword1", "keyword2", ...],
    "time_filter": "last_1_year", "last_3_years", "last_5_years" or null,
    "must_include": [],
    "exclude": [],
    "expected_count": 20
}}

Example:
Query: "Latest advances in reinforcement learning"
Output: {{
    "primary_goal": "survey",
    "keywords": ["reinforcement learning", "deep reinforcement learning"],
    "time_filter": "last_3_years",
    "must_include": [],
    "exclude": [],
    "expected_count": 20
}}

Important: Output ONLY valid JSON.
"""


def get_search_strategy_prompt(search_plan: dict, iteration: int) -> str:
    return f"""
Based on the following query analysis results, formulate a specific search strategy:

Analysis Results: {search_plan}
Current Iteration: {iteration}

Consider:
1. If primary_goal is "survey", prioritize papers containing "survey", "review", "overview"
2. If time_filter is specified, set appropriate time ranges
3. Adjust search parameters based on iteration count (e.g., expand keywords, relax time range)

Output JSON format search strategy:
{{
    "sources": ["semantic_scholar", "arxiv"],
    "query_params": {{
        "semantic_scholar": {{
            "query": "search query string",
            "max_results": 20,
            "year_filter": "optional time filter"
        }},
        "arxiv": {{
            "query": "search query string",
            "max_results": 20
        }}
    }},
    "max_results_per_source": 20,
    "prioritize_surveys": true or false,
    "iteration": {iteration}
}}
"""


def get_refinement_prompt(search_plan: dict, quality_metrics: dict, iteration: int) -> str:
    return f"""
Based on previous search results quality, adjust the search strategy:

Original Search Plan: {search_plan}
Quality Assessment Results: {quality_metrics}
Current Attempt: {iteration + 1} (already tried {iteration} times)

Problem Analysis: {quality_metrics.get('issues', [])}
Suggestions: {quality_metrics.get('suggestions', [])}

Adjust the search strategy to improve:
1. If breadth is insufficient, expand keyword range or add data sources
2. If depth is insufficient, raise quality threshold or adjust ranking weights
3. If timeliness is insufficient, adjust time filtering

Output the adjusted JSON format search strategy (same format as before).
"""


def get_quality_assessment_prompt(papers: list, search_strategy: dict) -> str:
    # 提取前5篇论文标题用于评估
    paper_titles = [p.get("title", "Unknown")[:50] + "..." for p in papers[:5]]

    return f"""
Assess the quality of current search results:

Search Strategy: {search_strategy}
Number of Papers Returned: {len(papers)}
Top 5 Papers: {paper_titles}

Please score from the following dimensions (0-1):
1. Breadth: Coverage of key concepts and field aspects
2. Depth: Overall quality of papers (citations, venue quality, survey nature)
3. Timeliness: Compliance with time requirements

Output in JSON format:
{{
    "breadth_score": 0.0,
    "depth_score": 0.0,
    "timeliness_score": 0.0,
    "overall_score": 0.0,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}
"""


def get_report_generation_prompt(query: str, search_plan: dict,
                                 search_strategy: dict, papers: list,
                                 quality_metrics: dict) -> str:
    # 提取前3篇代表性论文
    top_papers = papers[:3]
    paper_details = "\n".join([
        f"{i + 1}. {p.get('title', 'Unknown')[:60]}... (Score: {p.get('relevance_score', 0):.2f})"
        for i, p in enumerate(top_papers)
    ])

    return f"""
Generate a professional search report based on the following search process:

Original Query: {query}
Search Plan: {search_plan}
Search Strategy: {search_strategy}
Quality Assessment: {quality_metrics}
Total Papers Found: {len(papers)}

Representative Papers:
{paper_details}

Please generate a concise 2-3 sentence explanation explaining:
1. Search focus (e.g., whether focused on surveys, timeliness, high impact, etc.)
2. Main characteristics of returned results
3. Any notes (e.g., limited coverage in certain areas)

Use Chinese for the explanation, maintain professional but friendly tone.
"""