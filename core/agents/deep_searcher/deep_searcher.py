import asyncio
import os
from typing import Any, Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string
)
from prompts import get_checklist_gen_prompt
from state import (
    Checklist,
    DeepSearcherState,
    DeepSearcherOutput,
    ConductSearch,
    SearchComplete
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from core.utils.logs import logger
from core.agents.deep_searcher.configuration import DeepSearcherConfig


configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "temperature", "model_provider"),
)
logger.info("Configurable model initialized.")


async def generate_checklist(state: DeepSearcherState, config: RunnableConfig) -> Command[Literal["conduct_search"]]:
    configurable = DeepSearcherConfig.from_runnable_config(config)
    logger.info(f"Configurable: {configurable}")
    
    if not configurable.add_checklist:
        return Command(
            goto="conduct_search",
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
        goto="conduct_search",
        update={"checklist": response}
    )


async def conduct_search(state: DeepSearcherState, config: RunnableConfig) -> Command[Literal["search_web"]]:
    configurable = DeepSearcherConfig.from_runnable_config(config)

    search_tools = [ConductSearch, SearchComplete]

    conduct_search_model_config = {
        "model": configurable.conduct_search_model,
        "max_tokens": 10000,
        "temperature": 0.0,
        "timeout": 120
    }

    conduct_search_model = (
        configurable_model
        .bind_tools(search_tools)
        .with_config(conduct_search_model_config)
    )

    # response = await conduct_search_model.ainvoke([HumanMessage(content="Conduct a search for the following queries: " + ", ".join(state["search_queries"]))])
    import random
    if random.random() < 0.5:
        tool_call_obj = ConductSearch(queries=["What is the capital of France?", "What is the population of France?"])
        # 创建 AIMessage 并添加 tool_calls
        response = AIMessage(
            content="",
            tool_calls=[{
                "name": "ConductSearch",
                "args": tool_call_obj.model_dump(),
                "id": "test-id"
            }]
        )
    else:
        tool_call_obj = SearchComplete()
        response = AIMessage(
            content="",
            tool_calls=[{
                "name": "SearchComplete",
                "args": {},
                "id": "test-id"
            }]
        )

    return Command(
        goto="search_web",
        update={
            "searcher_messages": [response],
            "search_turns": state.get("search_turns", 0) + 1
        }
    )

async def search_web(state: DeepSearcherState, config: RunnableConfig) -> Command[Literal["conduct_search", END]]:
    configurable = DeepSearcherConfig.from_runnable_config(config)

    search_turns = state["search_turns"]
    most_recent_message = state["searcher_messages"][-1]

    exceeded_allowed_search_turns = search_turns > configurable.max_search_turns
    no_tool_calls = not most_recent_message.tool_calls
    
    search_complete_tool_call = any(
        tool_call["name"] == "SearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_allowed_search_turns or no_tool_calls or search_complete_tool_call:
        return Command(
            goto=END
        )

    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]

    logger.info(f"Conduct research calls: {conduct_research_calls}")
    return Command(
        goto="conduct_search"
    )


async def summarize_search_results(state: DeepSearcherState, config: RunnableConfig) -> Command[Literal[END]]:
    pass


checklist_node = StateGraph(DeepSearcherState)
checklist_node.add_node("generate_checklist", generate_checklist)
checklist_node.add_node("conduct_search", conduct_search)
checklist_node.add_node("search_web", search_web)

checklist_node.add_edge(START, "generate_checklist")
checklist_node.add_edge("generate_checklist", END)

checklist_graph = checklist_node.compile()


async def main():
    state = DeepSearcherState(messages=[HumanMessage(content="What is the capital of France?")])
    result = await checklist_graph.ainvoke(state)
    logger.info(str(result))

if __name__ == "__main__":
    asyncio.run(main())