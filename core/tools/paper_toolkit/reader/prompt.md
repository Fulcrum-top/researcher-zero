# 知识框架构建 Prompt 设计

## 核心目标

从多篇论文（特别是Survey类论文）中提取和构建一个结构化的领域知识框架，用于指导后续论文阅读和知识图谱构建。

## Prompt 内容

你是一位资深的学术研究专家，现在需要分析{papers_count}篇论文并从中构建一个结构化的领域知识框架。

论文内容如下：
{papers_text}

请根据这些论文内容，构建一个清晰、系统的领域知识框架。要求如下：

1. 框架应涵盖这些论文涉及的主要概念、方法、技术和研究方向
2. 展现概念之间的关系和层级结构
3. 识别当前研究的热点和前沿
4. 指出存在的研究空白和未来可能的发展方向

请严格按照以下JSON格式输出：

```json
{
  "framework_overview": {
    "domain_scope": "领域范围描述",
    "research_status": "当前研究状况概述",
    "development_trend": "发展趋势分析"
  },
  "core_concepts": [
    {
      "concept": "核心概念名称",
      "definition": "概念定义",
      "importance": "重要性说明",
      "relationships": [
        {
          "related_concept": "相关概念",
          "relationship_type": "关系类型（如包含、依赖、对立等）"
        }
      ]
    }
  ],
  "methodologies": [
    {
      "method": "方法名称",
      "description": "方法描述",
      "application_scenarios": "应用场景",
      "advantages": "优势",
      "limitations": "局限性"
    }
  ],
  "research_frontiers": [
    {
      "frontier": "研究前沿",
      "description": "详细描述",
      "significance": "意义"
    }
  ],
  "knowledge_graph_suggestion": {
    "nodes": [
      {
        "id": "节点ID",
        "name": "节点名称",
        "type": "节点类型（概念/方法/问题等）",
        "importance": "重要性评分（0-1）"
      }
    ],
    "edges": [
      {
        "source": "起始节点ID",
        "target": "目标节点ID",
        "relationship": "关系描述"
      }
    ]
  }
}
```

确保输出是合法的JSON格式，不要包含其他内容。

## 输出结构说明

### framework_overview - 框架概览
- `domain_scope`: 描述该知识框架适用的领域范围
- `research_status`: 对当前研究状况的总体评价
- `development_trend`: 领域发展的趋势预测

### core_concepts - 核心概念
- `concept`: 核心概念的名称
- `definition`: 概念的准确定义
- `importance`: 该概念在领域中的重要性说明
- `relationships`: 与其他概念的关系列表

### methodologies - 方法学
- `method`: 研究方法或技术的名称
- `description`: 方法的详细描述
- `application_scenarios`: 适用的应用场景
- `advantages`: 方法的优势
- `limitations`: 方法的局限性

### research_frontiers - 研究前沿
- `frontier`: 研究前沿方向的名称
- `description`: 详细描述
- `significance`: 该前沿方向的重要意义

### knowledge_graph_suggestion - 知识图谱建议
- `nodes`: 知识图谱中的节点信息
- `edges`: 节点间的关系信息