# core/tools/paper_toolkit/__init__.py
from typing import List, Dict, Any
from . import search
from .reader import reader

def get_tools() -> List[Dict[str, Any]]:
    """返回此工具包提供的所有工具信息列表"""
    # 初始化搜索工具包装器
    search_wrapper = search.PaperSearchMCPWrapper()
    # 初始化阅读工具包装器
    reader_wrapper = reader.PaperReaderMCPWrapper()
    
    # 获取所有工具定义
    tools = []
    tools.extend(search_wrapper.get_tools())
    tools.extend(reader_wrapper.get_tools())
    
    return tools