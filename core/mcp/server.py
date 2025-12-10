"""
MCP 主服务器 - 支持多种传输方式
"""
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import uvicorn

from core.mcp.models import ToolCallRequest, ToolCallResult, ToolSchema
from core.mcp.registry import registry

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP服务器核心"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.app = FastAPI(title="ResearcherZero MCP Server")
        self._setup_routes()

    def _setup_routes(self):
        """设置HTTP和WebSocket路由"""

        # HTTP 接口
        @self.app.get("/tools")
        async def list_tools() -> List[Dict[str, Any]]:
            """列出所有可用工具"""
            tools = registry.list_tools()
            return [tool.dict() for tool in tools]

        @self.app.post("/tools/{tool_name}")
        async def call_tool(tool_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
            """调用工具 (HTTP模式)"""
            try:
                result = await self._execute_tool(tool_name, request)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"工具调用失败 {tool_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # WebSocket 接口 (支持标准MCP协议)
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    # 读取消息
                    message = await websocket.receive_json()

                    # 处理工具调用
                    if message.get("method") == "tools/call":
                        request = ToolCallRequest(**message.get("params", {}))
                        result = await self._execute_tool(request.name, request.arguments)

                        # 发送结果
                        response = {
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}
                        }
                        await websocket.send_json(response)

                    # 处理工具列表请求
                    elif message.get("method") == "tools/list":
                        tools = registry.list_tools()
                        response = {
                            "jsonrpc": "2.0",
                            "id": message.get("id"),
                            "result": {"tools": [t.dict() for t in tools]}
                        }
                        await websocket.send_json(response)

            except WebSocketDisconnect:
                logger.info("WebSocket 连接断开")
            except Exception as e:
                logger.error(f"WebSocket 错误: {e}")
                await websocket.close(code=1011)

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """执行工具"""
        tool = registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"工具不存在: {tool_name}")

        logger.info(f"执行工具: {tool_name}, 参数: {arguments}")
        return await tool.executor(**arguments)

    def run(self, debug: bool = False):
        """运行服务器"""
        logger.info(f"启动 MCP 服务器: http://{self.host}:{self.port}")
        logger.info(f"工具数量: {len(registry.list_tools())}")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if debug else "info"
        )


# ============ 工具包注册函数 ============
def register_paper_tools():
    """注册论文工具包"""
    from core.tools.paper_toolkit import get_tools
    from core.mcp.registry import registry
    
    tools = get_tools()
    for tool in tools:
        registry.register_tool(tool)
    logger.info("论文工具包注册完成")


def register_all_tools():
    """注册所有工具包（项目启动时调用）"""
    # 注册论文工具
    register_paper_tools()

    # 未来可以在这里注册其他工具包
    # register_web_tools()
    # register_memory_tools()

    logger.info(f"所有工具注册完成，共 {len(registry.list_tools())} 个工具")
