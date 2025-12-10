"""
文件操作工具
"""
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """文件操作工具类"""

    @staticmethod
    def save_markdown(content: str, filename: str, directory: str = "data/knowledge") -> str:
        """
        保存Markdown文件

        设计理念：Markdown是人类和机器都可读的中间表示
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

        filepath = Path(directory) / filename

        # 添加时间戳和元信息
        metadata = f"""---
created: {datetime.now().isoformat()}
type: structured_knowledge
---
"""

        full_content = metadata + content

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)

        logger.info(f"Saved markdown to {filepath}")
        return str(filepath)

    @staticmethod
    def save_json(data: Dict[str, Any], filename: str, directory: str = "data/cache") -> str:
        """保存JSON文件"""
        Path(directory).mkdir(parents=True, exist_ok=True)

        filepath = Path(directory) / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    @staticmethod
    def load_json(filepath: str) -> Optional[Dict[str, Any]]:
        """加载JSON文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON {filepath}: {e}")
            return None

    @staticmethod
    def generate_paper_filename(paper_title: str, arxiv_id: str) -> str:
        """生成论文文件名"""
        # 清理标题
        clean_title = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in paper_title)
        clean_title = clean_title[:100]  # 限制长度

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{clean_title}_{arxiv_id}_{timestamp}.md"

    @staticmethod
    def generate_knowledge_filename(domain: str, concept: str = "") -> str:
        """生成知识文件名"""
        clean_domain = ''.join(c if c.isalnum() else '_' for c in domain.lower())
        if concept:
            clean_concept = ''.join(c if c.isalnum() else '_' for c in concept.lower())
            return f"knowledge_{clean_domain}_{clean_concept}_{datetime.now().strftime('%Y%m%d')}.md"
        else:
            return f"knowledge_{clean_domain}_overview_{datetime.now().strftime('%Y%m%d')}.md"