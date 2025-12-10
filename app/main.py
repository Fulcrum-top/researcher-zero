# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi_mcp import FastApiMCP
import logging

from core.mcp.registry import registry
from core.tools.paper_toolkit import get_tools as get_paper_tools

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：注册所有工具包
    logger.info("正在注册工具包...")
    # 注册论文工具包
    registry.register_toolkit(get_paper_tools)
    # 未来可以在这里注册其他工具包
    # registry.register_toolkit(get_web_tools)
    yield
    # 关闭时：清理资源
    logger.info("应用关闭，清理资源...")

# 创建FastAPI应用
app = FastAPI(title="ResearcherZero", lifespan=lifespan)

# 创建并挂载MCP服务器
# FastAPI-MCP会自动扫描已注册的工具并暴露[citation:2][citation:10]
mcp = FastApiMCP(
    app,
    name="ResearcherZero Tools",
    description="为研究型AI智能体提供的工具集。"
)
mcp.mount()  # 默认挂载在 /mcp 端点[citation:1]

@app.get("/")
async def root():
    return {"message": "ResearcherZero API 服务已运行", "mcp_endpoint": "/mcp"}

@app.get("/tools")
async def list_registered_tools():
    """查看所有已注册的工具（调试用）"""
    return registry.list_tools()