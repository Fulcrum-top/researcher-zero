"""
MCP 核心数据模型
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum

# ============ 协议模型 (MCP规范) ============
class ToolSchema(BaseModel):
    """MCP工具定义"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class ToolCallRequest(BaseModel):
    """工具调用请求"""
    name: str
    arguments: Dict[str, Any]

class ToolCallResult(BaseModel):
    """工具调用结果"""
    content: List[Dict[str, Any]]
    isError: bool = False

# ============ 内部注册模型 ============
@dataclass
class RegisteredTool:
    """内部注册的工具实例"""
    schema: ToolSchema
    executor: Callable[..., Awaitable[Any]]

class SearchMode(str, Enum):
    """搜索模式枚举"""
    SURVEY = "survey"
    GENERAL = "general"