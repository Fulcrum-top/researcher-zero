import os

import pytest

from core.agents.deep_searcher.configuration import DeepSearcherConfig


def test_from_runnable_config_prefers_environment(monkeypatch):
    """环境变量优先于 runnable config，同名字段会覆盖."""
    monkeypatch.setenv("ADD_CHECKLIST", "false")
    monkeypatch.setenv("MAX_SEARCH_TURNS", "10")
    config = {
        "configurable": {
            "add_checklist": True,
            "max_search_turns": 3,
            "max_search_results": 8,
        }
    }

    result = DeepSearcherConfig.from_runnable_config(config)

    assert result.add_checklist is False
    assert result.max_search_turns == 10
    assert result.max_search_results == 8


def test_from_runnable_config_uses_config_when_env_absent():
    """当缺少环境变量时，使用 runnable config 中的值."""
    config = {
        "configurable": {
            "max_search_results": 12,
            "summarization_model": "custom-model",
        }
    }

    result = DeepSearcherConfig.from_runnable_config(config)

    assert result.max_search_results == 12
    assert result.summarization_model == "custom-model"


def test_from_runnable_config_falls_back_to_defaults(monkeypatch):
    """环境与配置都缺失时，使用默认值初始化."""
    monkeypatch.delenv("ADD_CHECKLIST", raising=False)
    monkeypatch.delenv("CHECKLIST_GEN_MODEL", raising=False)

    result = DeepSearcherConfig.from_runnable_config(None)

    assert result.add_checklist is True
    assert result.checklist_gen_model == "kimi-k2-0905-preview"
