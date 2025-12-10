# core/mcp/registry.py
from typing import Dict, Any, List, Callable, Optional
import logging
from core.tools.base import BaseTool

logger = logging.getLogger(__name__)

class ToolRegistry:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def register_tool(self, tool_info: Dict[str, Any]):
        """向全局注册一个工具"""
        name = tool_info.get("name")
        self._tools[name] = tool_info
        logger.info(f"工具注册成功: {name}")

    def register_toolkit(self, toolkit_module):
        """注册一个工具包（模块）"""
        try:
            # 约定：工具包模块需要有一个 `get_tools()` 函数
            tools_to_register = toolkit_module.get_tools()
            for tool_info in tools_to_register:
                self.register_tool(tool_info)
            logger.info(f"工具包注册成功: {toolkit_module.__name__}")
        except AttributeError as e:
            logger.error(f"工具包 '{toolkit_module.__name__}' 缺少 `get_tools()` 函数: {e}")
            raise

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return list(self._tools.values())

# 全局单例
registry = ToolRegistry()