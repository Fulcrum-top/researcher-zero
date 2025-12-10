# def main():
#     print("Hello from researcher-zero!")
# if __name__ == "__main__":
#     main()

# ================================================================

# !/usr/bin/env python3
"""
ResearcherZero Learning 模块主运行文件 - 集成版本
"""
import argparse
import logging
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(Path(__file__).parent.parent.as_posix())

# 移除了对本地gateway的依赖，使用外部统一LLM网关
from core.agents.learning_agent import LearningAgent
from core.configs.settings import settings
from core.utils.logs import logger

def list_available_models():
    """列出可用模型"""
    # 由于使用外部LLM网关，这里简单列出一些常用模型
    models = ["kimi-latest", "deepseek-chat", "qwen-turbo", "gpt-3.5-turbo", "gpt-4"]
    
    logger.info("可用模型列表:")
    logger.info("=" * 60)

    for model in models:
        logger.info(f"\n🔹 {model}")

    logger.info("\n" + "=" * 60)
    logger.info("提示: 使用 --model 参数指定要使用的模型")

def validate_model(model_name: str) -> bool:
    """验证模型是否可用"""
    # 由于使用外部LLM网关，这里简单检查模型名称是否合理
    # 实际验证需要通过调用网关来完成
    if not model_name:
        logger.error("模型名称不能为空")
        return False
    
    # 检查是否设置了必要的环境变量
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        logger.error("请设置 OPENAI_BASE_URL 环境变量指向外部LLM网关")
        return False
        
    logger.info(f"模型 {model_name} 验证通过（使用外部网关: {base_url}）")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ResearcherZero Learning Module - 学术领域知识提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --domain "reinforcement learning"
  %(prog)s --domain "computer vision" --model deepseek-chat
  %(prog)s --domain "自然语言处理" --output-dir ./my_knowledge
        """
    )

    parser.add_argument("--domain", type=str, default=settings.default_domain,
                        help="要学习的研究领域（默认：artificial intelligence）")
    parser.add_argument("--output-dir", type=str, default=settings.storage.knowledge_dir,
                        help="知识输出目录")
    parser.add_argument("--model", type=str, default="kimi-latest",
                        help=f"使用的模型（默认：kimi-latest）")
    parser.add_argument("--list-models", action="store_true",
                        help="列出可用模型并退出")
    parser.add_argument("--top-k", type=int, default=settings.tools.top_k_papers,
                        help=f"处理的Top K论文数（默认：{settings.tools.top_k_papers}）")
    parser.add_argument("--max-results", type=int, default=settings.arxiv.max_results,
                        help=f"搜索最大结果数（默认：{settings.arxiv.max_results}）")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细日志")

    args = parser.parse_args()

    # 设置详细日志
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 列出模型并退出
    if args.list_models:
        list_available_models()
        return

    # 验证模型
    if not validate_model(args.model):
        logger.info("模型验证失败，请使用 --list-models 查看可用模型")
        return

    # 更新配置
    settings.tools.top_k_papers = args.top_k
    settings.arxiv.max_results = args.max_results
    settings.storage.knowledge_dir = args.output_dir

    logger.info("启动 ResearcherZero Learning 模块")
    logger.info(f"学习领域: {args.domain}")
    logger.info(f"使用模型: {args.model}")
    logger.info(f"处理论文数: Top {args.top_k}")
    logger.info(f"搜索数量: {args.max_results} 篇")
    logger.info(f"输出目录: {args.output_dir}")

    # 创建Agent
    try:
        agent = LearningAgent(model_name=args.model)
    except Exception as e:
        logger.error(f"Agent初始化失败: {e}")
        return

    # 执行学习
    logger.info(f"\n开始学习: {args.domain}")
    logger.info("=" * 60)

    result = agent.learn_domain(args.domain)

    # 输出结果
    logger.info("\n" + "=" * 60)
    if result["success"]:
        logger.info("学习完成！")
        logger.info(f"理论文: {result['papers_processed']} 篇")
        logger.info(f"知识单元: {result['knowledge_units']} 个")
        logger.info(f"总用时: {result['total_time']:.2f} 秒")

        if result.get("knowledge_path"):
            logger.info(f"知识文档: {result['knowledge_path']}")
        if result.get("state_path"):
            logger.info(f"状态文件: {result['state_path']}")
    else:
        logger.info("学习失败")
        logger.info(f"错误信息: {result.get('error', '未知错误')}")

    return result


if __name__ == "__main__":
    # 确保日志目录存在
    Path("logs").mkdir(exist_ok=True)

    main()