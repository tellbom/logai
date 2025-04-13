# utils/logger.py
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback
from typing import Optional, Dict, Any, List, Union

from config import get_config


def setup_logging(
        log_level: str = None,
        log_dir: str = None,
        app_name: str = None,
        console: bool = True,
        file: bool = True,
        max_bytes: int = 10485760,  # 10 MB
        backup_count: int = 10,
        format_string: str = None,
        date_format: str = '%Y-%m-%d %H:%M:%S'
) -> logging.Logger:
    """
    设置日志系统

    Args:
        log_level: 日志级别，可以是'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        log_dir: 日志目录
        app_name: 应用名称，用于日志文件名
        console: 是否输出到控制台
        file: 是否输出到文件
        max_bytes: 日志文件最大字节数
        backup_count: 保留的备份文件数量
        format_string: 日志格式字符串
        date_format: 日期格式字符串

    Returns:
        根日志记录器
    """
    # 加载配置
    config = get_config()

    # 设置默认值
    if log_level is None:
        log_level = config.get("app.log_level", "INFO")

    if log_dir is None:
        log_dir = config.get("app.log_dir", "./logs")

    if app_name is None:
        app_name = config.get("app.name", "log-ai-analyzer")

    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"

    # 确保日志目录存在
    if file and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 解析日志级别
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # 清除现有处理器（避免重复）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建格式化器
    formatter = logging.Formatter(format_string, date_format)

    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 添加文件处理器
    if file:
        log_file = os.path.join(log_dir, f"{app_name}.log")

        # 创建按大小轮转的文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # 创建按日期轮转的文件处理器（每天一个文件）
        daily_log_file = os.path.join(log_dir, f"{app_name}_%Y%m%d.log")
        daily_handler = TimedRotatingFileHandler(
            daily_log_file,
            when='midnight',
            interval=1,
            backupCount=30,  # 保留30天
            encoding='utf-8'
        )
        daily_handler.setFormatter(formatter)
        root_logger.addHandler(daily_handler)

    # 设置第三方库的日志级别
    third_party_loggers = [
        "elasticsearch",
        "urllib3",
        "qdrant_client",
        "sentence_transformers"
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # 设置模块自定义日志级别
    module_log_levels = config.get("logging.modules", {})
    for module_name, level in module_log_levels.items():
        module_level = getattr(logging, level.upper(), None)
        if module_level:
            logging.getLogger(module_name).setLevel(module_level)

    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，级别: {log_level}, 日志目录: {log_dir}")

    return root_logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    定制的日志适配器，添加上下文信息到日志
    """

    def __init__(self, logger, extra=None):
        """
        初始化适配器

        Args:
            logger: 原始日志记录器
            extra: 额外的上下文信息
        """
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        """
        处理日志消息，添加额外信息
        """
        # 添加上下文前缀
        context_items = []
        for key, value in self.extra.items():
            if value is not None:
                context_items.append(f"{key}={value}")

        # 如果有上下文信息，添加到消息前面
        if context_items:
            context_str = "[" + " ".join(context_items) + "]"
            msg = f"{context_str} {msg}"

        return msg, kwargs


def get_logger(
        name: str,
        context: Dict[str, Any] = None,
        default_level: str = None
) -> Union[logging.Logger, LoggerAdapter]:
    """
    获取日志记录器，可选择性地添加上下文

    Args:
        name: 日志记录器名称
        context: 上下文信息
        default_level: 默认日志级别

    Returns:
        日志记录器或适配器
    """
    logger = logging.getLogger(name)

    # 设置默认级别
    if default_level:
        level = getattr(logging, default_level.upper(), None)
        if level is not None:
            logger.setLevel(level)

    # 如果提供了上下文，返回适配器
    if context:
        return LoggerAdapter(logger, context)

    return logger


def log_exception(
        logger: Union[logging.Logger, LoggerAdapter],
        exc: Exception,
        message: str = "发生异常",
        level: str = "ERROR"
) -> None:
    """
    记录异常信息

    Args:
        logger: 日志记录器
        exc: 异常对象
        message: 自定义消息
        level: 日志级别
    """
    log_func = getattr(logger, level.lower(), logger.error)

    # 获取完整的堆栈跟踪
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_text = ''.join(tb_lines)

    # 记录异常
    log_func(f"{message}: {str(exc)}\n{tb_text}")


def get_performance_logger(name: str, context: Dict[str, Any] = None) -> 'PerformanceLogger':
    """
    获取性能日志记录器

    Args:
        name: 日志记录器名称
        context: 上下文信息

    Returns:
        性能日志记录器
    """
    logger = get_logger(name, context)
    return PerformanceLogger(logger)


class PerformanceLogger:
    """
    性能日志记录器，用于记录操作执行时间
    """

    def __init__(self, logger: Union[logging.Logger, LoggerAdapter]):
        """
        初始化性能日志记录器

        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.start_times = {}

    def start(self, operation: str):
        """
        开始计时操作

        Args:
            operation: 操作名称
        """
        self.start_times[operation] = time.time()

    def end(self, operation: str, level: str = "INFO", additional_info: Dict[str, Any] = None):
        """
        结束计时操作并记录

        Args:
            operation: 操作名称
            level: 日志级别
            additional_info: 额外信息
        """
        if operation not in self.start_times:
            self.logger.warning(f"操作 '{operation}' 未启动计时")
            return

        elapsed = time.time() - self.start_times[operation]
        del self.start_times[operation]

        # 构建日志消息
        msg = f"操作 '{operation}' 完成，耗时: {elapsed:.4f}秒"

        if additional_info:
            info_str = ", ".join(f"{k}={v}" for k, v in additional_info.items())
            msg += f" ({info_str})"

        # 记录日志
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(msg)

    def __enter__(self):
        """
        上下文管理器入口
        """
        self.start('context_operation')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器退出
        """
        if exc_type is None:
            self.end('context_operation')
        else:
            elapsed = time.time() - self.start_times.get('context_operation', time.time())
            self.logger.error(f"操作失败，耗时: {elapsed:.4f}秒，异常: {exc_val}")
            return False  # 不抑制异常


# 全局初始化
def initialize_logging() -> logging.Logger:
    """
    初始化全局日志系统

    Returns:
        根日志记录器
    """
    config = get_config()

    log_level = config.get("app.log_level", "INFO")
    log_dir = config.get("app.log_dir", "./logs")
    app_name = config.get("app.name", "log-ai-analyzer")

    return setup_logging(log_level, log_dir, app_name)