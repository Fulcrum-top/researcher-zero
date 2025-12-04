from typing import List


def get_system_prompt() -> str:
    return """You are a helpful deep searcher.
"""

def get_checklist_gen_prompt(query: str) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query cannot be an empty string")

    instruction = """Your task is to generate a checklist that guides the upcoming information search.
<Checklist Design Principles>
1. Control the granularity
- Each checklist item should cover one "independently searchable dimension" rather than fragmented facts.
- ✓ Good example: "Does it cover every known navigation mechanism type (magnetoreception, celestial cues, olfaction, etc.)?"
- ✗ Bad example: "Does it mention magnetoreception?" "Does it mention olfaction?" (too granular)

2. Verifiability
- Every question must allow a clear present/absent judgment.
- Avoid vague prompts like "Is it detailed enough?"
- Prefer checklist items such as "Does it explain the biological mechanism of X (receptors, neural pathways)?"

3. Dual-track structure
**Comprehensiveness Checks**
[Goal]: Ensure horizontal coverage across the topic.
[Focus]:
- Are all core components of the topic represented?
- Are the relevant external contexts and background factors included?
- Are variations across scenarios/species/cases considered?

**Depth Checks**
[Goal]: Ensure vertical analytical depth.
[Focus]:
- Does it explain the underlying mechanisms (biological/physical/cognitive)?
- Does it analyse system integration (multi-factor interactions, hierarchy, calibration)?
- Does it evaluate evidence quality (controversies, open questions, methodological limits)?
- Does it connect to ecological or applied significance (adaptive value, conservation implications)?
</Checklist Design Principles>

<Generation Workflow>
Step 1: Decompose the query
- Identify the core phenomena/process/problem.
- Surface the implicit information needs (how/why/which factors).

Step 2: Enumerate dimensions
- List the "must-cover subdomains/aspects" for comprehensiveness.
- List the "analysis layers that require deep investigation" for depth.

Step 3: Turn dimensions into checklist items
- Create one concise question per dimension.
- Start each question with "Does it" to target the search goal directly.
</Generation Workflow>

<Output Requirements>
- Comprehensiveness Checks must have at most 5 items.
- Depth Checks must have at most 4 items.
- Each checklist item must be independent, concise, and actionable.
</Output Requirements>"""

    return f"Given query: {normalized_query}\n\n{instruction}"


def get_deep_searcher_think_prompt(info_task: str, checklist: str, history_queries: List[str], history_search_results: str) -> str:
    checklist_guide_prompt = ""
    if checklist:
        checklist_guide_prompt = f"""<Checklist Guide>
Use the checklist as a coverage map. Identify uncovered or shallow items. Decide to:
1) Continue to conduct searches for the uncovered or shallow items with the ConductSearch tool.
2) Complete if all checklist dimensions are sufficiently addressed.
</Checklist Guide>

**Checklist**
{checklist}
"""
    history_queries_prompt = ""
    if history_queries:
        history_queries_prompt = f"""<History Queries>
You have already conducted multiple searches with the following queries:
{history_queries}
</History Queries>
"""

    return f"""<Task>
Your job is to use tools to gather information about a information seeking task. You can use any of the tools provided to you to find resources that can help answer the question. You can call these tools in series or in parallel.
</Task>

{checklist_guide_prompt}

{history_queries_prompt}

<History Search Results>
{history_search_results}
</History Search Results>

<Information Seeking Task>
{info_task}
</Information Seeking Task>

<Instructions>
Think like a human researcher with limited time. Follow these steps:
1. **Read the task carefully** - What specific information does the task need?
2. **Generate diverse queries** - Each query must explore a DIFFERENT aspect, angle, or dimension:
   - For simple factual questions (e.g., "What is X?"): Use ONE comprehensive query, avoid repeating the same question in different words
   - For complex topics: Each query should target a different subdomain, perspective, or analytical layer
   - GOOD examples for "How do birds navigate?":
     * "bird navigation mechanisms magnetoreception"
     * "bird migration patterns routes"
     * "avian celestial navigation cues"
   - BAD examples (too similar, avoid these):
     * "capital of France"
     * "what is the capital city of France"
     * "France capital city name"
3. **Max queries per turn** - Generate at most 3 queries per search turn to maintain focus and efficiency.
4. **No duplicate searches** - Do not repeat any query that has been used before, and avoid queries that ask essentially the same question with different wording.
5. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
6. **Execute narrower searches as you gather information** - Fill in the gaps with targeted queries
7. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>
"""


def get_summarization_prompt(info_task: str, checklist: str, history_search_results: str) -> str:
    checklist_prompt = ""
    if checklist:
        checklist_prompt = f"""<Checklist>
The checklist is a guide for the summary. It is used to ensure that all relevant information is gathered.
{checklist}
</Checklist>
"""
    return f"""<Task>
Your job is to summarize the web search results gathered about the information seeking task.
</Task>

<Search Results>
{history_search_results}
</Search Results>

<Information Seeking Task>
{info_task}
</Information Seeking Task>

{checklist_prompt}

<Instructions>
1. **Information Extraction:**
   - Carefully review all the search results from multiple searches
   - Identify and extract factual information that is relevant to the information seeking task
   - Focus on information that directly addresses the task

2. **Content Organization:**
   - Organize the extracted information in a logical and coherent manner
   - Prioritize the most relevant and important information
   - Ensure that the summary is comprehensive and covers all the information that is relevant to the task
</Instructions>
"""