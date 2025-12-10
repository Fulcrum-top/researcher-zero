"""
学习Agent - 基于LangGraph的单一Agent实现
更新：集成LLMAdapter和LearningState
"""
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import time

from langgraph.graph import StateGraph, END

from core.infra.learning_state import (
    LearningStateManager, PaperMetadata, ProcessedPaper, KnowledgeUnit
)
# 移除了对本地gateway的依赖，使用外部统一LLM网关
from core.tools.paper_tools import PaperTools
from core.tools.search_tools import SearchTools
from core.utils.file_utils import FileUtils
from core.configs.settings import settings
from utils.logs import logger

# 添加外部LLM网关的调用方式
import litellm
import os


class LearningAgent:
    """学习Agent - 集成版本"""

    def __init__(self, model_name: str = "kimi-latest"):
        # 使用外部统一LLM网关，通过环境变量配置base_url和api_key
        self.model_name = model_name
        # 确保设置了外部LLM网关的base_url
        if not os.environ.get("OPENAI_BASE_URL"):
            raise ValueError("请设置OPENAI_BASE_URL环境变量指向外部LLM网关")
        
        # 初始化工具
        self.paper_tools = PaperTools()
        self.search_tools = SearchTools()
        self.file_utils = FileUtils()

        # 状态管理器
        self.state_manager: Optional[LearningStateManager] = None

        # 构建工作流
        self.workflow = self._build_workflow()
        logger.info(f"LearningAgent initialized with model: {model_name}")

    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""

        def route_after_search(state):
            """搜索后的路由"""
            if not state["papers_found"]:
                return "handle_error"
            return "select_top_papers"

        def route_after_selection(state):
            """选择后的路由"""
            if not state["papers_selected"]:
                return "handle_error"
            return "process_next_paper"

        def route_after_paper_processing(state):
            """论文处理后的路由"""
            if state["current_paper_index"] < len(state["papers_selected"]):
                return "process_next_paper"
            return "integrate_knowledge"

        # 定义节点函数
        nodes = {
            "initialize": lambda state: self._initialize(state),
            "search_surveys": lambda state: self._search_surveys(state),
            "select_top_papers": lambda state: self._select_top_papers(state),
            "process_next_paper": lambda state: self._process_next_paper(state),
            "analyze_paper": lambda state: self._analyze_paper(state),
            "extract_references": lambda state: self._extract_references(state),
            "integrate_knowledge": lambda state: self._integrate_knowledge(state),
            "save_results": lambda state: self._save_results(state),
            "handle_error": lambda state: self._handle_error(state)
        }

        # 构建图
        workflow = StateGraph(Dict[str, Any])

        for name, func in nodes.items():
            workflow.add_node(name, func)

        # 设置边
        workflow.set_entry_point("initialize")

        workflow.add_edge("initialize", "search_surveys")
        workflow.add_conditional_edges(
            "search_surveys",
            route_after_search,
            {
                "select_top_papers": "select_top_papers",
                "handle_error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "select_top_papers",
            route_after_selection,
            {
                "process_next_paper": "process_next_paper",
                "handle_error": "handle_error"
            }
        )
        workflow.add_edge("process_next_paper", "analyze_paper")
        workflow.add_edge("analyze_paper", "extract_references")
        workflow.add_conditional_edges(
            "extract_references",
            route_after_paper_processing,
            {
                "process_next_paper": "process_next_paper",
                "integrate_knowledge": "integrate_knowledge"
            }
        )
        workflow.add_edge("integrate_knowledge", "save_results")
        workflow.add_edge("save_results", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def _initialize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """初始化步骤"""
        domain = state.get("domain", settings.default_domain)

        # 创建状态管理器
        self.state_manager = LearningStateManager(domain)

        # 记录初始决策
        self.state_manager.record_decision(
            decision="start_learning",
            rationale=f"开始学习领域: {domain}",
            context={"model": self.llm_adapter.model_name}
        )

        return {
            **state,
            "step": "initialized",
            "state_manager": self.state_manager
        }

    def _search_surveys(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """搜索Survey论文"""
        try:
            domain = state["domain"]
            logger.info(f"搜索领域: {domain}")

            # 使用工具搜索论文
            papers = self.paper_tools.search_papers(
                query=f"{domain} survey",
                max_results=settings.arxiv.max_results
            )

            # 添加到状态
            for paper in papers:
                self.state_manager.add_paper(paper)

            # 记录决策
            self.state_manager.record_decision(
                decision="select_search_strategy",
                rationale=f"搜索{len(papers)}篇相关论文",
                context={"query": f"{domain} survey", "results_count": len(papers)}
            )

            return {
                **state,
                "step": "surveys_searched",
                "papers_found": papers
            }

        except Exception as e:
            error_msg = f"搜索失败: {str(e)}"
            self.state_manager.update_state(error=error_msg)
            logger.error(error_msg)
            return state

    def _select_top_papers(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """选择Top K论文"""
        try:
            papers = state["papers_found"]

            if not papers:
                raise ValueError("没有找到论文")

            # 选择策略：基于相关性分数和Survey关键词
            def paper_score(paper: PaperMetadata) -> float:
                score = paper.relevance_score

                # Survey论文加分
                title_lower = paper.title.lower()
                abstract_lower = paper.abstract.lower()

                survey_keywords = ["survey", "review", "overview", "comprehensive"]
                for keyword in survey_keywords:
                    if keyword in title_lower or keyword in abstract_lower:
                        score += 0.5
                        break

                # 近期论文加分（简化处理）
                if "2024" in paper.published or "2023" in paper.published:
                    score += 0.3

                return score

            # 排序并选择
            sorted_papers = sorted(papers, key=paper_score, reverse=True)
            selected_papers = sorted_papers[:settings.tools.top_k_papers]

            # 更新状态
            self.state_manager.update_state(papers_selected=selected_papers)

            # 记录决策
            self.state_manager.record_decision(
                decision="select_top_papers",
                rationale=f"从{len(papers)}篇中选择{len(selected_papers)}篇进行深度分析",
                context={
                    "selection_criteria": "relevance_score + survey_keywords + recency",
                    "top_k": settings.tools.top_k_papers
                }
            )

            return {
                **state,
                "step": "papers_selected",
                "papers_selected": selected_papers,
                "current_paper_index": 0
            }

        except Exception as e:
            error_msg = f"论文选择失败: {str(e)}"
            self.state_manager.update_state(error=error_msg)
            logger.error(error_msg)
            return state

    def _process_next_paper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理下一篇论文"""
        try:
            selected_papers = state["papers_selected"]
            current_index = state.get("current_paper_index", 0)

            if current_index >= len(selected_papers):
                return state  # 所有论文已处理

            paper = selected_papers[current_index]
            logger.info(f"处理论文 {current_index + 1}/{len(selected_papers)}: {paper.title}")

            # 下载论文
            start_time = time.time()
            pdf_path = self.paper_tools.download_paper(paper.arxiv_id)
            download_time = time.time() - start_time

            if not pdf_path:
                logger.warning(f"无法下载论文: {paper.arxiv_id}")
                # 跳过这篇论文
                return {
                    **state,
                    "current_paper_index": current_index + 1,
                    "current_paper": None
                }

            # 提取内容
            content = self.paper_tools.extract_paper_content(pdf_path)

            # 更新状态
            self.state_manager.update_state(
                papers_downloaded=self.state_manager.state["papers_downloaded"] + 1
            )

            return {
                **state,
                "step": "paper_downloaded",
                "current_paper_index": current_index,
                "current_paper": {
                    "metadata": paper,
                    "pdf_path": pdf_path,
                    "content": content,
                    "download_time": download_time
                }
            }

        except Exception as e:
            error_msg = f"论文处理失败: {str(e)}"
            self.state_manager.update_state(error=error_msg)
            logger.error(error_msg)
            return state

    def _analyze_paper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """分析论文"""
        try:
            current_paper = state.get("current_paper")
            if not current_paper:
                # 跳过分析，直接进入下一篇
                return {
                    **state,
                    "current_paper_index": state["current_paper_index"] + 1
                }

            paper_meta = current_paper["metadata"]
            content = current_paper["content"]

            logger.info(f"分析论文: {paper_meta.title}")

            # 使用LLM分析论文
            start_time = time.time()
            analysis = self.llm_adapter.analyze_paper(
                paper_title=paper_meta.title,
                abstract=paper_meta.abstract,
                content=content.get("full_text", "")[:3000]  # 限制长度
            )
            analysis_time = time.time() - start_time

            # 创建ProcessedPaper
            processed_paper = ProcessedPaper(
                metadata=paper_meta,
                content_preview=content.get("full_text", "")[:1000],
                structured_analysis=analysis,
                references=content.get("references", []),
                concepts=analysis.get("concepts", []),
                processed_at=datetime.now().isoformat()
            )

            # 添加到状态
            self.state_manager.add_processed_paper(processed_paper)

            # 提取知识单元
            for concept in analysis.get("concepts", []):
                knowledge_unit = KnowledgeUnit(
                    concept=concept,
                    definition=f"来自论文: {paper_meta.title}",
                    sources=[paper_meta.arxiv_id],
                    relationships=[],
                    confidence=0.8,  # 初始置信度
                    created_at=datetime.now().isoformat()
                )
                self.state_manager.add_knowledge_unit(knowledge_unit)

            # 记录推理轨迹
            self.state_manager.add_reasoning_trace({
                "paper": paper_meta.title,
                "analysis": analysis,
                "analysis_time": analysis_time,
                "timestamp": datetime.now().isoformat()
            })

            return {
                **state,
                "step": "paper_analyzed",
                "analysis_time": analysis_time
            }

        except Exception as e:
            error_msg = f"论文分析失败: {str(e)}"
            self.state_manager.update_state(error=error_msg)
            logger.error(error_msg)
            return state

    def _extract_references(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """提取参考文献（已在上一步完成）"""
        # 参考文献已在analyze_paper步骤中提取
        current_index = state.get("current_paper_index", 0) + 1

        return {
            **state,
            "step": "references_extracted",
            "current_paper_index": current_index
        }

    def _integrate_knowledge(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """整合知识"""
        try:
            processed_papers = self.state_manager.state["papers_processed"]

            if not processed_papers:
                raise ValueError("没有已处理的论文")

            logger.info(f"整合{len(processed_papers)}篇论文的知识")

            # 准备论文信息
            papers_info = []
            for paper in processed_papers:
                papers_info.append({
                    "title": paper.metadata.title,
                    "analysis": paper.structured_analysis
                })

            # 使用LLM整合知识
            start_time = time.time()
            integrated_knowledge = self.llm_adapter.integrate_knowledge(
                papers_info=papers_info,
                domain=state["domain"]
            )
            integration_time = time.time() - start_time

            # 更新状态
            self.state_manager.update_state(
                structured_knowledge=integrated_knowledge,
                processing_time=self.state_manager.state["processing_time"] + integration_time
            )

            # 记录决策
            self.state_manager.record_decision(
                decision="integrate_knowledge",
                rationale=f"整合{len(processed_papers)}篇论文的知识",
                context={
                    "integration_time": integration_time,
                    "knowledge_units": len(self.state_manager.state["knowledge_units"])
                }
            )

            return {
                **state,
                "step": "knowledge_integrated",
                "integration_time": integration_time,
                "integrated_knowledge": integrated_knowledge
            }

        except Exception as e:
            error_msg = f"知识整合失败: {str(e)}"
            self.state_manager.update_state(error=error_msg)
            logger.error(error_msg)
            return state

    def _save_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """保存结果"""
        try:
            domain = state["domain"]
            logger.info(f"保存{domain}领域的学习结果")

            # 1. 保存状态
            state_filename = f"learning_state_{domain.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            state_path = f"{settings.storage.knowledge_dir}/states/{state_filename}"
            self.state_manager.save_state(state_path)

            # 2. 生成并保存知识文档
            knowledge_content = self._generate_knowledge_document()
            knowledge_filename = f"knowledge_{domain.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md"
            knowledge_path = self.file_utils.save_markdown(
                knowledge_content,
                knowledge_filename,
                settings.storage.knowledge_dir
            )

            # 3. 保存每篇论文的详细文档
            for paper in self.state_manager.state["papers_processed"]:
                paper_content = self._generate_paper_document(paper)
                paper_filename = self.file_utils.generate_paper_filename(
                    paper.metadata.title,
                    paper.metadata.arxiv_id
                )
                self.file_utils.save_markdown(
                    paper_content,
                    paper_filename,
                    f"{settings.storage.knowledge_dir}/papers"
                )

            # 4. 保存性能报告
            performance_report = self._generate_performance_report()
            report_filename = f"report_{domain.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.file_utils.save_json(
                performance_report,
                report_filename,
                f"{settings.storage.knowledge_dir}/reports"
            )

            logger.info(f"结果已保存到: {knowledge_path}")

            return {
                **state,
                "step": "results_saved",
                "knowledge_path": knowledge_path,
                "state_path": state_path
            }

        except Exception as e:
            error_msg = f"结果保存失败: {str(e)}"
            self.state_manager.update_state(error=error_msg)
            logger.error(error_msg)
            return state

    def _generate_knowledge_document(self) -> str:
        """生成知识文档"""
        state = self.state_manager.state

        md = f"""# {state['domain']} 领域知识报告

## 概览
- **学习时间**: {state['start_time']}
- **处理论文数**: {len(state['papers_processed'])}
- **知识单元数**: {len(state['knowledge_units'])}
- **总用时**: {state['processing_time']:.2f}秒

## 领域概览
"""

        # 添加结构化知识
        if state.get("structured_knowledge"):
            knowledge = state["structured_knowledge"]

            if knowledge.get("overview"):
                overview = knowledge["overview"]
                md += f"\n### 研究背景\n{overview.get('background', '')}\n"
                md += f"\n### 现状分析\n{overview.get('current_state', '')}\n"
                md += f"\n### 发展趋势\n{overview.get('trends', '')}\n"

            if knowledge.get("concept_system"):
                md += "\n## 核心概念体系\n"
                for concept in knowledge["concept_system"][:10]:  # 限制数量
                    if isinstance(concept, dict):
                        md += f"\n- **{concept.get('concept', '')}**: {concept.get('definition', '')}"
                    else:
                        md += f"\n- {concept}"

        # 添加知识单元
        md += "\n\n## 详细知识单元\n"
        for unit in state["knowledge_units"][:20]:  # 限制数量
            md += unit.to_markdown() + "\n\n"

        # 添加论文列表
        md += "\n## 分析的基础论文\n"
        for i, paper in enumerate(state["papers_processed"], 1):
            md += f"\n{i}. **{paper.metadata.title}**\n"
            md += f"   作者: {', '.join(paper.metadata.authors)}\n"
            md += f"   arXiv ID: {paper.metadata.arxiv_id}\n"
            md += f"   核心概念: {', '.join(paper.concepts[:3]) if paper.concepts else '无'}\n"

        # 添加决策记录
        if state["decisions"]:
            md += "\n## 关键决策记录\n"
            for decision in state["decisions"]:
                md += f"\n### {decision['decision']}\n"
                md += f"**依据**: {decision['rationale']}\n"
                md += f"**时间**: {decision['timestamp']}\n"

        return md

    def _generate_paper_document(self, paper: ProcessedPaper) -> str:
        """生成论文文档"""
        md = f"""# {paper.metadata.title}

## 元数据
- **作者**: {', '.join(paper.metadata.authors)}
- **发布时间**: {paper.metadata.published}
- **arXiv ID**: {paper.metadata.arxiv_id}
- **分类**: {', '.join(paper.metadata.categories)}
- **相关性分数**: {paper.metadata.relevance_score:.2f}

## 摘要
{paper.metadata.abstract}

## 结构化分析
"""

        # 添加分析结果
        analysis = paper.structured_analysis
        if isinstance(analysis, dict):
            if analysis.get("contributions"):
                md += "\n### 核心贡献\n"
                for contribution in analysis["contributions"]:
                    md += f"- {contribution}\n"

            if analysis.get("concepts"):
                md += "\n### 关键概念\n"
                for concept in analysis["concepts"]:
                    md += f"- {concept}\n"

            if analysis.get("findings"):
                md += "\n### 主要发现\n"
                for finding in analysis["findings"]:
                    md += f"- {finding}\n"
        else:
            md += f"\n{analysis}\n"

        # 添加参考文献
        if paper.references:
            md += f"\n## 参考文献 ({len(paper.references)}篇)\n"
            for ref in paper.references[:10]:  # 限制数量
                md += f"\n- {ref.get('authors', '')} ({ref.get('year', '')}) {ref.get('title', '')}\n"

        return md

    def _generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        state = self.state_manager.state

        return {
            "domain": state["domain"],
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "papers_found": len(state["papers_found"]),
                "papers_processed": len(state["papers_processed"]),
                "knowledge_units": len(state["knowledge_units"]),
                "tokens_used": state["tokens_used"],
                "processing_time": state["processing_time"],
                "papers_downloaded": state["papers_downloaded"]
            },
            "decisions": state["decisions"],
            "state_summary": self.state_manager.get_state_summary()
        }

    def _handle_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """错误处理"""
        error_msg = state.get("error", "未知错误")
        logger.error(f"工作流错误: {error_msg}")

        # 保存错误状态
        error_report = {
            "domain": state.get("domain", "unknown"),
            "step": state.get("step", "unknown"),
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "state_summary": self.state_manager.get_state_summary() if self.state_manager else {}
        }

        self.file_utils.save_json(
            error_report,
            f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            f"{settings.storage.knowledge_dir}/errors"
        )

        return state

    def learn_domain(self, domain: str) -> Dict[str, Any]:
        """学习特定领域"""
        initial_state = {
            "domain": domain,
            "step": "start",
            "papers_found": [],
            "papers_selected": [],
            "current_paper_index": 0,
            "error": None
        }

        try:
            # 执行工作流
            start_time = time.time()
            final_state = self.workflow.invoke(initial_state)
            total_time = time.time() - start_time

            # 获取结果
            if self.state_manager:
                state_summary = self.state_manager.get_state_summary()

                result = {
                    "success": self.state_manager.state["error"] is None,
                    "domain": domain,
                    "papers_processed": state_summary["papers_processed"],
                    "knowledge_units": state_summary["knowledge_units"],
                    "tokens_used": state_summary["tokens_used"],
                    "total_time": total_time,
                    "final_step": final_state.get("step", ""),
                    "error": self.state_manager.state["error"],
                    "knowledge_path": final_state.get("knowledge_path"),
                    "state_path": final_state.get("state_path")
                }
            else:
                result = {
                    "success": False,
                    "domain": domain,
                    "error": "State manager not initialized"
                }

            logger.info(f"学习完成: {result}")
            return result

        except Exception as e:
            logger.error(f"学习工作流异常: {e}")
            return {
                "success": False,
                "domain": domain,
                "error": str(e)
            }