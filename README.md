# researcher-zero
## 启动
用 uv run

## 日志系统
项目使用统一的全局 logger，自动识别调用模块。

### 使用方式
```python
from core.utils.logs import logger

logger.info("信息日志")
logger.debug("调试日志")
logger.warning("警告日志")
logger.error("错误日志")
```

**重要：请不要使用 `print()`，统一使用 logger。**

### 配置
通过环境变量 `VERBOSE` 控制日志启用/禁用：
- `VERBOSE=true` 或 `VERBOSE=1`：启用（默认）
