"""
MCP 客户端 - 供 LangGraph Agent 使用
"""
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from core.mcp.models import ToolSchema

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 协议客户端"""

    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        self.server_url = server_url
        self.ws_url = server_url.replace("http", "ws") + "/ws"
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        """连接服务器"""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()
            self.session = None

    async def list_tools(self) -> List[ToolSchema]:
        """获取可用工具列表"""
        await self.connect()
        async with self.session.get(f"{self.server_url}/tools") as resp:
            data = await resp.json()
            return [ToolSchema(**item) for item in data]

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """调用工具"""
        await self.connect()
        async with self.session.post(
                f"{self.server_url}/tools/{tool_name}",
                json=kwargs
        ) as resp:
            data = await resp.json()
            if data.get("success"):
                return data.get("result")
            else:
                raise Exception(f"工具调用失败: {data.get('detail')}")

    async def create_langgraph_tools(self) -> List[BaseTool]:
        """
        自动创建 LangGraph 工具
        动态从服务器获取工具列表并生成对应的 Tool 对象
        """
        tools = await self.list_tools()
        langgraph_tools = []

        for tool_schema in tools:
            # 动态创建输入模型
            input_fields = {}
            for prop_name, prop_info in tool_schema.inputSchema.get("properties", {}).items():
                field_type = self._map_json_schema_to_pydantic(prop_info.get("type", "string"))
                description = prop_info.get("description", "")
                required = prop_name in tool_schema.inputSchema.get("required", [])

                if required:
                    input_fields[prop_name] = (field_type, Field(..., description=description))
                else:
                    default = prop_info.get("default")
                    input_fields[prop_name] = (field_type, Field(default=default, description=description))

            # 动态创建 Pydantic 模型
            InputModel = type(
                f"{tool_schema.name.capitalize()}Input",
                (BaseModel,),
                {"__annotations__": {k: v[0] for k, v in input_fields.items()},
                 **{k: v[1] for k, v in input_fields.items()}}
            )

            # 创建 LangGraph Tool
            tool = type(
                f"{tool_schema.name.capitalize()}Tool",
                (BaseTool,),
                {
                    "name": tool_schema.name,
                    "description": tool_schema.description,
                    "args_schema": InputModel,
                    "mcp_client": self,
                    "_run": lambda self, **kwargs: self._execute_tool_sync(**kwargs),
                    "_arun": lambda self, **kwargs: self._execute_tool_async(**kwargs),
                }
            )()

            langgraph_tools.append(tool)

        logger.info(f"创建了 {len(langgraph_tools)} 个 LangGraph 工具")
        return langgraph_tools

    async def _execute_tool_async(self, **kwargs):
        """异步执行工具"""
        return await self.call_tool(self.name, **kwargs)

    def _execute_tool_sync(self, **kwargs):
        """同步执行工具（包装异步调用）"""
        import asyncio
        return asyncio.run(self._execute_tool_async(**kwargs))

    @staticmethod
    def _map_json_schema_to_pydantic(json_type: str):
        """映射 JSON Schema 类型到 Python 类型"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return type_mapping.get(json_type, str)