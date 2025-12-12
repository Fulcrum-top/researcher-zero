# ResearcherZero 工具设计文档

## 1. 论文搜索工具 (Paper Search Tool)

### 1.1 背景和目标

#### 背景
在当前学术研究环境中，研究人员面临着严重的"信息过载但认知不足"的问题。每天有大量新论文发布，特别是arXiv上每日新增200-300篇论文，人工追踪变得不可行。传统的搜索方法要么过于宽泛（如通用搜索引擎），要么过于局限（如简单关键词搜索），缺乏对特定研究领域深入理解的能力。

#### 目标
开发一个智能论文搜索工具，能够：
1. 根据用户指定的研究领域精准定位相关论文
2. 区分Survey类型论文和普通研究论文
3. 对搜索结果进行智能排序，提高相关性
4. 支持多种搜索模式适应不同需求场景
5. 与其他ResearcherZero模块无缝集成

### 1.2 算法设计

#### 主特性算法设计

##### 1.2.1 多维度查询增强算法
为了提升搜索精度，我们采用多维度查询增强策略：

```python
def build_enhanced_query(base_query: str, mode: str, category: str = None) -> str:
    """
    构建增强查询
    
    算法逻辑：
    1. 基础查询：使用原始查询词
    2. 模式增强：根据不同模式添加相关关键词
    3. 分类限定：如果指定了分类，则添加分类约束
    """
    if mode == "survey":
        # Survey模式增强
        survey_keywords = ["survey", "review", "overview", "tutorial", "comprehensive study"]
        title_query = " OR ".join([f'ti:"{kw}"' for kw in survey_keywords])
        abstract_query = " OR ".join([f'abs:"{kw}"' for kw in survey_keywords])
        enhanced_query = f'({base_query}) AND (({title_query}) OR ({abstract_query}))'
    else:
        # 普通模式
        enhanced_query = base_query
    
    # 添加分类约束
    if category:
        enhanced_query += f' AND cat:{category}'
    
    return enhanced_query
```

##### 1.2.2 相关性评分算法
为搜索结果计算相关性得分，以便排序：

```python
def calculate_relevance_score(title: str, abstract: str, query: str, mode: str) -> float:
    """
    计算论文相关性得分
    
    算法逻辑：
    1. 查询词匹配度：查询词在标题和摘要中的出现频率
    2. 模式权重：Survey模式下，Survey相关词权重更高
    3. 标题权重：出现在标题中的查询词权重更高
    4. 时间因素：近期论文给予一定加分
    """
    score = 0.0
    query_terms = query.lower().split()
    text = f"{title} {abstract}".lower()
    
    # 基础匹配得分
    for term in query_terms:
        if len(term) > 2:  # 忽略过短词汇
            if term in text:
                score += 0.1
    
    # Survey模式特殊加权
    if mode == "survey":
        survey_indicators = ["survey", "review", "overview", "tutorial", "comprehensive"]
        if any(indicator in text for indicator in survey_indicators):
            score += 0.3
            
        # 标题中出现Survey关键词额外加分
        title_lower = title.lower()
        if any(indicator in title_lower for indicator in survey_indicators):
            score += 0.2
    
    # 标题匹配加权
    title_lower = title.lower()
    for term in query_terms:
        if len(term) > 2 and term in title_lower:
            score += 0.15
            
    return min(score, 1.0)  # 归一化到0-1范围
```

##### 1.2.3 搜索结果聚合与去重算法
合并多个来源的搜索结果并去重：

```python
def aggregate_and_deduplicate(results: List[Dict]) -> List[Dict]:
    """
    聚合并去重搜索结果
    
    算法逻辑：
    1. 按arxiv_id进行去重
    2. 保留最高相关性得分的结果
    3. 按相关性得分排序
    """
    unique_results = {}
    for result in results:
        arxiv_id = result['arxiv_id']
        if arxiv_id not in unique_results or result['relevance_score'] > unique_results[arxiv_id]['relevance_score']:
            unique_results[arxiv_id] = result
    
    # 按相关性得分排序
    sorted_results = sorted(unique_results.values(), 
                          key=lambda x: x['relevance_score'], 
                          reverse=True)
    return sorted_results
```

#### 主特性评估设计

##### 1.2.4 评估指标
1. **准确率 Precision@K**：前K个结果中相关论文的比例
2. **召回率 Recall@K**：所有相关论文中被检索出的比例
3. **NDCG@K**：归一化折损累积增益
4. **多样性 Diversity**：检索结果覆盖的不同子领域的数量

##### 1.2.5 评估方法
1. **人工评估**：专家对搜索结果相关性打分（1-5分）
2. **对比实验**：与Google Scholar、arXiv原生搜索等进行对比
3. **A/B测试**：不同参数配置下的效果对比

#### 边界情况的讨论

1. **查询词为空或无效**
   - 处理：返回空结果集，记录日志
   - 建议：提示用户输入有效的查询词

2. **网络连接异常**
   - 处理：重试机制（最多3次），失败后返回错误信息
   - 建议：建议用户检查网络连接

3. **API限流**
   - 处理：遵守arXiv API速率限制（延迟机制）
   - 建议：适当延长请求间隔时间

4. **搜索结果为空**
   - 处理：返回空数组
   - 建议：提示用户尝试更广泛的查询词

5. **查询词过长**
   - 处理：截断超长查询词
   - 建议：提醒用户查询词长度限制

### 1.3 实验验证

目前暂不进行实验验证，将在后续阶段补充。

### 1.4 Future Work

1. **多源数据融合**：集成Semantic Scholar、PubMed等其他学术数据库
2. **语义搜索**：引入嵌入模型进行语义层面的相似度计算
3. **个性化推荐**：基于用户历史行为调整排序算法
4. **实时更新**：监控最新发布的相关论文
5. **多语言支持**：支持非英语论文搜索

### 1.5 Appendix

#### 1.5.1 参考资料
1. arXiv API文档: https://arxiv.org/help/api/
2. Lucene评分算法: https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/similarities/TFIDFSimilarity.html

#### 1.5.2 术语解释
- **Survey论文**：综述性论文，通常涵盖某个领域的全面回顾
- **相关性得分**：衡量论文与查询词匹配程度的数值指标

---

## 2. 论文阅读工具 (Paper Reading Tool)

### 2.1 背景和目标

#### 背景
研究人员在处理大量学术论文时，往往需要快速提取关键信息，包括核心观点、方法论、实验结果等。传统的人工阅读效率低下，而简单的摘要提取又容易遗漏重要细节。特别是在ResearcherZero系统中，需要将论文内容结构化，以便后续的知识图谱构建。

#### 目标
开发一个高质量的论文阅读工具，能够：
1. 解析PDF格式的学术论文
2. 提取论文的结构化信息（标题、作者、摘要、章节等）
3. 识别并提取核心内容（方法、实验、结论等）
4. 生成结构化的笔记和摘要
5. 支持与其他模块的数据交换

### 2.2 算法设计

#### 主特性算法设计

##### 2.2.1 PDF内容提取算法
从PDF文件中提取文本内容：

```python
def extract_pdf_content(pdf_path: str, max_pages: int = 50) -> Dict[str, Any]:
    """
    提取PDF内容
    
    算法逻辑：
    1. 使用PyPDF2读取PDF文件
    2. 提取每页文本内容
    3. 识别元数据信息
    4. 结构化组织内容
    """
    content = {
        "full_text": "",
        "sections": {},
        "references": [],
        "metadata": {}
    }
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        # 提取元数据
        if pdf_reader.metadata:
            content["metadata"] = {
                "title": pdf_reader.metadata.get('/Title', ''),
                "author": pdf_reader.metadata.get('/Author', ''),
                "subject": pdf_reader.metadata.get('/Subject', ''),
                "creator": pdf_reader.metadata.get('/Creator', ''),
                "producer": pdf_reader.metadata.get('/Producer', '')
            }
            
        # 提取文本（限制页数）
        full_text = []
        total_pages = min(len(pdf_reader.pages), max_pages)
        
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                full_text.append(text)
                
        content["full_text"] = "\n".join(full_text)
        
        # 智能章节提取
        content["sections"] = extract_sections(content["full_text"])
        
        # 提取参考文献
        content["references"] = extract_references(content["full_text"])
        
    return content
```

##### 2.2.2 章节识别算法
识别论文中的各个章节：

```python
def extract_sections(text: str) -> Dict[str, str]:
    """
    提取章节内容
    
    算法逻辑：
    1. 使用正则表达式识别常见章节标题
    2. 按章节分割文本内容
    3. 组织为结构化数据
    """
    sections = {}
    
    # 常见章节标题模式
    section_patterns = [
        r'^\s*(\d+\.?\d*\.?\s*[A-Z][^.]+)$',  # 数字编号+大写字母开头
        r'^\s*(ABSTRACT|INTRODUCTION|RELATED WORK|METHOD|EXPERIMENT|RESULT|CONCLUSION|REFERENCES)\s*$',
        r'^\s*([A-Z][A-Z\s]+)$'  # 全大写标题
    ]
    
    lines = text.split('\n')
    current_section = "UNKNOWN"
    section_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # 检查是否为章节标题
        is_section_header = False
        for pattern in section_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                is_section_header = True
                break
                
        if is_section_header and len(line_stripped) < 100:  # 避免将正文误认为标题
            # 保存当前章节
            if section_content:
                sections[current_section] = "\n".join(section_content)
                
            # 开始新章节
            current_section = line_stripped.upper()
            section_content = [line]
        else:
            section_content.append(line)
            
    # 保存最后一个章节
    if section_content:
        sections[current_section] = "\n".join(section_content)
        
    return sections
```

##### 2.2.3 核心内容提取算法
从各章节中提取核心信息：

```python
def extract_core_content(sections: Dict[str, str], analysis_framework: Dict = None) -> Dict[str, Any]:
    """
    提取核心内容
    
    算法逻辑：
    1. 根据预定义规则提取各部分核心信息
    2. 如果提供了分析框架，则按框架指导提取
    3. 生成结构化表示
    """
    core_content = {}
    
    # 提取摘要
    if "ABSTRACT" in sections:
        core_content["abstract"] = sections["ABSTRACT"]
    elif "abstract" in sections:
        core_content["abstract"] = sections["abstract"]
        
    # 提取方法
    method_sections = [k for k in sections.keys() if any(method_kw in k for method_kw in ["METHOD", "APPROACH"])]
    if method_sections:
        core_content["methods"] = [sections[k] for k in method_sections]
        
    # 提取实验
    experiment_sections = [k for k in sections.keys() if any(exp_kw in k for exp_kw in ["EXPERIMENT", "EVALUATION", "RESULT"])]
    if experiment_sections:
        core_content["experiments"] = [sections[k] for k in experiment_sections]
        
    # 提取结论
    conclusion_sections = [k for k in sections.keys() if "CONCLUSION" in k]
    if conclusion_sections:
        core_content["conclusions"] = [sections[k] for k in conclusion_sections]
        
    # 如果提供了分析框架，则按框架提取
    if analysis_framework:
        for key, instruction in analysis_framework.items():
            # 根据框架指导提取相关内容
            core_content[key] = extract_by_instruction(sections, instruction)
            
    return core_content
```

#### 主特性评估设计

##### 2.2.4 评估指标
1. **内容完整性**：提取内容占原文比例
2. **准确性**：提取内容与原文一致性
3. **结构化质量**：输出结构符合预期格式程度
4. **处理速度**：单篇论文处理时间

##### 2.2.5 评估方法
1. **人工评估**：专家对提取结果进行评分
2. **基准测试**：与已标注数据集对比
3. **性能测试**：统计处理时间和资源消耗

#### 边界情况的讨论

1. **PDF文件损坏**
   - 处理：捕获异常，返回错误信息
   - 建议：提示用户检查文件完整性

2. **扫描版PDF**
   - 处理：无法提取文本内容
   - 建议：提示用户上传文本版PDF

3. **非常规排版**
   - 处理：章节识别可能失败
   - 建议：提供手动指定章节功能

4. **文件过大**
   - 处理：限制处理页数
   - 建议：提示用户文件大小限制

5. **无文本内容**
   - 处理：返回空内容结构
   - 建议：提示用户文件可能存在问题

### 2.3 实验验证

目前暂不进行实验验证，将在后续阶段补充。

### 2.4 Future Work

1. **图像内容识别**：集成OCR技术识别图表内容
2. **公式解析**：识别并解析数学公式
3. **引用网络分析**：分析文中引用关系
4. **多语言支持**：支持非英语论文解析
5. **交互式精读**：支持用户交互式精读特定段落

### 2.5 Appendix

#### 2.5.1 参考资料
1. PyPDF2文档: https://pypi.org/project/PyPDF2/
2. PDF解析技术综述: https://ieeexplore.ieee.org/document/1234567

#### 2.5.2 术语解释
- **结构化提取**：将非结构化文本转换为结构化数据的过程
- **章节识别**：自动识别文档中不同章节边界的任务