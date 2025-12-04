import os
from typing import Any
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class DeepSearcherConfig(BaseModel):
    # Pipeline parameters
    add_checklist: bool = Field(
        default=True,
        description="Whether to add a generated checklist to guide the search"
    )
    max_think_turns: int = Field(
        default=1,
        description="The maximum number of search turns to perform"
    )
    max_search_results: int = Field(
        default=5,
        description="The maximum number of search results to return"
    )
    # Model parameters
    checklist_gen_model: str = Field(
        default="kimi",
        description="The model to use for generating the checklist"
    )
    deep_searcher_think_model: str = Field(
        default="kimi",
        description="The model to use for generating the search queries"
    )
    summarization_model: str = Field(
        default="kimi",
        description="The model to use for summarizing the search results"
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "DeepSearcherConfig":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})