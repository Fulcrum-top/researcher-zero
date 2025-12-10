"""
基于LLM的论文阅读工具实现
该工具通过调用LLM从多篇论文中总结出结构化知识框架
"""
from typing import List, Dict, Any, Optional
# 更改导入，使用外部统一LLM网关
import litellm
import os


class PaperReaderTool:
    """基于LLM的论文阅读工具"""

    def __init__(self, model_name: str = "kimi-latest"):
        """
        初始化工具
        
        Args:
            model_name: 使用的LLM模型名称
        """
        self.model_name = model_name
        # 确保设置了外部LLM网关的base_url
        if not os.environ.get("OPENAI_BASE_URL"):
            raise ValueError("请设置OPENAI_BASE_URL环境变量指向外部LLM网关")

    async def build_knowledge_framework(
        self,
        papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        从多篇论文中构建领域知识框架
        
        Args:
            papers: 论文信息列表，每篇论文应包含title、abstract、content等字段
            
        Returns:
            结构化的领域知识框架
        """
        # 构建输入文本
        papers_text = ""
        for i, paper in enumerate(papers, 1):
            papers_text += f"\n\n论文 {i}:\n"
            papers_text += f"标题: {paper.get('title', 'N/A')}\n"
            papers_text += f"摘要: {paper.get('abstract', 'N/A')}\n"
            papers_text += f"内容: {paper.get('content', '')[:2000]}...\n"  # 限制内容长度

        # 构建Prompt
        prompt = self._build_prompt(papers_text, len(papers))
        
        try:
            # 调用外部LLM网关
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = litellm.completion(
                model=self.model_name,
                messages=messages
            )
            
            response_text = response.choices[0].message.content
            
            # 解析响应
            return self._parse_response(response_text)
            
        except Exception as e:
            return {
                "error": f"构建知识框架时发生错误: {str(e)}",
                "papers_count": len(papers)
            }

    def _build_prompt(self, papers_text: str, papers_count: int) -> str:
        """
        构建用于生成知识框架的Prompt
        
        Args:
            papers_text: 所有论文的文本内容
            papers_count: 论文数量
            
        Returns:
            构造好的Prompt
        """
        prompt = f"""你是一位资深的学术研究专家，现在需要分析{papers_count}篇论文并从中构建一个结构化的领域知识框架。

论文内容如下：
{papers_text}

请根据这些论文内容，构建一个清晰、系统的领域知识框架。要求如下：

1. 框架应涵盖这些论文涉及的主要概念、方法、技术和研究方向
2. 展现概念之间的关系和层级结构
3. 识别当前研究的热点和前沿
4. 指出存在的研究空白和未来可能的发展方向

请严格按照以下JSON格式输出：

{{
  "framework_overview": {{
    "domain_scope": "领域范围描述",
    "research_status": "当前研究状况概述",
    "development_trend": "发展趋势分析"
  }},
  "core_concepts": [
    {{
      "concept": "核心概念名称",
      "definition": "概念定义",
      "importance": "重要性说明",
      "relationships": [
        {{
          "related_concept": "相关概念",
          "relationship_type": "关系类型（如包含、依赖、对立等）"
        }}
      ]
    }}
  ],
  "methodologies": [
    {{
      "method": "方法名称",
      "description": "方法描述",
      "application_scenarios": "应用场景",
      "advantages": "优势",
      "limitations": "局限性"
    }}
  ],
  "research_frontiers": [
    {{
      "frontier": "研究前沿",
      "description": "详细描述",
      "significance": "意义"
    }}
  ],
  "knowledge_graph_suggestion": {{
    "nodes": [
      {{
        "id": "节点ID",
        "name": "节点名称",
        "type": "节点类型（概念/方法/问题等）",
        "importance": "重要性评分（0-1）"
      }}
    ],
    "edges": [
      {{
        "source": "起始节点ID",
        "target": "目标节点ID",
        "relationship": "关系描述"
      }}
    ]
  }}
}}

确保输出是合法的JSON格式，不要包含其他内容。"""

        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM的响应
        
        Args:
            response: LLM的响应文本
            
        Returns:
            解析后的结构化数据
        """
        import json
        import re
        
        try:
            # 尝试提取JSON部分
            # 查找第一个{和最后一个}之间的内容
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # 如果没找到JSON格式，返回原始响应
                return {
                    "raw_response": response,
                    "parse_status": "failed"
                }
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON解析失败: {str(e)}",
                "raw_response": response
            }


# MCP工具包装器
class PaperReaderMCPWrapper:
    """论文阅读工具的MCP包装器"""

    def __init__(self):
        self.tool = PaperReaderTool()

    def get_tools(self) -> List[Dict[str, Any]]:
        """返回MCP工具定义"""
        return [
            {
                "name": "build_knowledge_framework",
                "description": "从多篇论文中构建结构化领域知识框架",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "papers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "abstract": {"type": "string"},
                                    "content": {"type": "string"}
                                },
                                "required": ["title", "abstract", "content"]
                            },
                            "description": "论文信息列表"
                        }
                    },
                    "required": ["papers"]
                }
            }
        ]

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        if tool_name == "build_knowledge_framework":
            return await self.tool.build_knowledge_framework(**kwargs)
        else:
            raise ValueError(f"未知工具: {tool_name}")