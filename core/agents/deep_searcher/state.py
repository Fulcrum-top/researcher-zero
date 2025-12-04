import operator
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import MessageLikeRepresentation


def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class Checklist(BaseModel):
    comprehensiveness: List[str] = Field(
        description="Comprehensiveness check questions"
    )
    depth: List[str] = Field(
        description="Depth check questions"
    )

class ConductSearch(BaseModel):
    """Call this tool to conduct multiple searches."""
    queries: List[str] = Field(
        description="List of search queries"
    )

class SearchComplete(BaseModel):
    """Call this tool to indicate that the search is complete."""

class SearchResult(BaseModel):
    url: str
    title: str
    content: str

class DeepSearcherState(MessagesState):
    """Main agent state containing messages and research data."""
    
    checklist: Optional[Checklist]
    search_turns: int = Field(default=0)
    search_results: List[SearchResult]
    search_queries: List[str]
    search_report: str

    searcher_messages: Annotated[list[MessageLikeRepresentation], override_reducer]

class DeepSearcherOutput(BaseModel):
    search_report: str