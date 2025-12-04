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

