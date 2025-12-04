import asyncio
from dotenv import load_dotenv
from typing import Literal, List
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

from core.agents.deep_searcher.prompts import (
    get_system_prompt,
    get_checklist_gen_prompt,
    get_deep_searcher_think_prompt,
    get_summarization_prompt
)
from core.agents.deep_searcher.state import (
    Checklist,
    DeepSearcherState,
    ConductSearch,
    SearchComplete,
    SearchResult
)
from core.utils.logs import logger
from core.agents.deep_searcher.configuration import DeepSearcherConfig
from core.tools.web_search.tavily_search import tavily_search

load_dotenv()
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "temperature", "model_provider"),
)
logger.info("Configurable model initialized.")


async def generate_checklist(state: DeepSearcherState, config: RunnableConfig) -> Command[Literal["deep_searcher_think"]]:
    configurable = DeepSearcherConfig.from_runnable_config(config)
    logger.info(f"Configurable: {configurable}")
    
    if not configurable.add_checklist:
        return Command(
            goto="deep_searcher_think",
            update={"checklist": ""}
        )

    messages = state["messages"]
    model_config = {
        "model": configurable.checklist_gen_model,
        "model_provider": "openai",
        "max_tokens": 10000,
        "temperature": 0.0,
        "timeout": 120
    }

    checklist_gen_model = configurable_model.with_structured_output(Checklist).with_config(model_config)
    checklist_gen_prompt = get_checklist_gen_prompt(get_buffer_string(messages))

    response = await checklist_gen_model.ainvoke([HumanMessage(content=checklist_gen_prompt)])
    return Command(
        goto="deep_searcher_think",
        update={"checklist": response}
    )


async def deep_searcher_think(state: DeepSearcherState, config: RunnableConfig) -> Command[Literal["deep_searcher_act"]]:
    configurable = DeepSearcherConfig.from_runnable_config(config)

    deep_searcher_think_model_config = {
        "model": configurable.deep_searcher_think_model,
        "model_provider": "openai",
        "max_tokens": 10000,
        "temperature": 0.0,
        "timeout": 120
    }

    actions = [ConductSearch, SearchComplete]
    deep_searcher_think_model = (
        configurable_model
        .bind_tools(actions)
        .with_config(deep_searcher_think_model_config)
    )

    checklist_value = state.get("checklist", "")
    checklist_str = checklist_value if isinstance(checklist_value, str) else str(checklist_value)
    history_search_queries = state.get("search_queries", [])
    history_search_results = state.get("search_results", [])
    history_search_results_prompt = "\n".join([f"{result.title}\n{result.content}" for result in history_search_results])
    deep_searcher_think_prompt = get_deep_searcher_think_prompt(
        state.get("info_task", ""),
        checklist_str,
        history_search_queries,
        history_search_results_prompt
    )

    messages = [SystemMessage(content=get_system_prompt()), HumanMessage(content=deep_searcher_think_prompt)] 
    response = await deep_searcher_think_model.ainvoke(messages)

    return Command(
        goto="deep_searcher_act",
        update={
            "searcher_messages": [response],
            "search_turns": state.get("search_turns", 0) + 1
        }
    )


async def deep_searcher_act(state: DeepSearcherState, config: RunnableConfig) -> Command[Literal["deep_searcher_think", "summarize"]]:
    configurable = DeepSearcherConfig.from_runnable_config(config)

    search_turns = state.get("search_turns", 0)
    searcher_messages = state.get("searcher_messages", [])
    if not searcher_messages:
        logger.warning("No searcher messages found; skipping to summarize.")
        return Command(goto="summarize")

    most_recent_message = searcher_messages[-1]

    exceeded_allowed_search_turns = search_turns >= configurable.max_search_turns
    no_tool_calls = not most_recent_message.tool_calls
    
    search_complete_tool_call = any(
        tool_call["name"] == "SearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_allowed_search_turns or no_tool_calls or search_complete_tool_call:
        return Command(
            goto="summarize"
        )

    conduct_search_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductSearch"
    ]
    logger.info(f"Conduct search task: {conduct_search_calls}")

    if not conduct_search_calls:
        return Command(goto="summarize")

    existing_queries = set(state.get("search_queries", []))
    aggregated_results: List[SearchResult] = []
    tool_messages: List[ToolMessage] = []
    new_queries: List[str] = []

    for tool_call in conduct_search_calls:
        queries = tool_call.get("args", {}).get("queries", []) or []
        unique_queries = [q for q in queries if q not in existing_queries]
        existing_queries.update(unique_queries)

        if not unique_queries:
            logger.info("All proposed queries were already searched; skipping.")
            continue

        logger.info(f"Running tavily search for queries: {unique_queries}")
        tavily_response = await tavily_search.ainvoke({
            "queries": unique_queries,
            "max_results": configurable.max_search_results
        })

        # tavily_response is a list aligned with queries input
        for query, query_results in zip(unique_queries, tavily_response):
            query_result_items = query_results.get("results", [])
            for item in query_result_items:
                aggregated_results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", "") or query,
                        content=item.get("content") or item.get("raw_content", "")
                    )
                )
            new_queries.append(query)

        tool_messages.append(
            ToolMessage(
                content=str(tavily_response),
                tool_call_id=tool_call["id"]
            )
        )

    if not aggregated_results:
        return Command(goto="summarize")

    updated_results = state.get("search_results", []) + aggregated_results
    updated_queries = state.get("search_queries", []) + new_queries

    return Command(
        goto="deep_searcher_think",
        update={
            "search_results": updated_results,
            "search_queries": updated_queries,
            "searcher_messages": tool_messages
        }
    )


async def summarize(state: DeepSearcherState, config: RunnableConfig) -> Command[Literal[END]]:
    configurable = DeepSearcherConfig.from_runnable_config(config)
    summarization_model_config = {
        "model": configurable.summarization_model,
        "model_provider": "openai",
        "max_tokens": 10000,
        "temperature": 0.0,
        "timeout": 180
    }
    summarization_model = configurable_model.with_config(summarization_model_config)
    checklist_value = state.get("checklist", "")
    checklist_str = checklist_value if isinstance(checklist_value, str) else str(checklist_value)
    history_search_results = "\n".join([f"{result.title}\n{result.content}" for result in state.get("search_results", [])])
    summarization_prompt = get_summarization_prompt(
        state.get("info_task", ""),
        checklist_str,
        history_search_results
    )
    messages = [SystemMessage(content=get_system_prompt()), HumanMessage(content=summarization_prompt)]
    response = await summarization_model.ainvoke(messages)
    return Command(
        goto=END,
        update={"search_report": response}
    )


deep_searcher_graph = StateGraph(DeepSearcherState)
deep_searcher_graph.add_node("generate_checklist", generate_checklist)
deep_searcher_graph.add_node("deep_searcher_think", deep_searcher_think)
deep_searcher_graph.add_node("deep_searcher_act", deep_searcher_act)
deep_searcher_graph.add_node("summarize", summarize)

deep_searcher_graph.add_edge(START, "generate_checklist")
deep_searcher_graph.add_edge("generate_checklist", "deep_searcher_think")
deep_searcher_graph.add_edge("deep_searcher_think", "deep_searcher_act")
deep_searcher_graph.add_edge("deep_searcher_act", "summarize")
deep_searcher_graph.add_edge("summarize", END)

deep_searcher_graph = deep_searcher_graph.compile()


async def main():
    sample_task = "深度调研 LifeLong Agent 的前沿技术"
    state = DeepSearcherState(
        messages=[HumanMessage(content=sample_task)],
        info_task=sample_task,
        checklist=None,
        search_turns=0,
        search_queries=[],
        search_results=[],
        search_report=None,
        searcher_messages=[]
    )
    result = await deep_searcher_graph.ainvoke(state)
    logger.info(str(result))
    breakpoint()

if __name__ == "__main__":
    asyncio.run(main())
