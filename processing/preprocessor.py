# processing/preprocessor.py
import re
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import json

logger = logging.getLogger(__name__)


class LogPreprocessor:
    """日志预处理器，用于清洗和标准化日志文本"""

    def __init__(
            self,
            time_pattern: str = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
            log_level_pattern: str = r"\[(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|ERR|TRACE)\]",
            exception_patterns: List[str] = [
                r"Exception:",
                r"Error:",
                r"Traceback \(most recent call last\):"
            ],
            max_stack_depth: int = 30,
            max_message_length: int = 10000
    ):
        """
        初始化日志预处理器

        Args:
            time_pattern: 时间戳正则表达式模式
            log_level_pattern: 日志级别正则表达式模式
            exception_patterns: 异常识别正则表达式模式列表
            max_stack_depth: 最大堆栈深度，用于提取堆栈跟踪
            max_message_length: 最大消息长度，超过将被截断
        """
        self.time_pattern = re.compile(time_pattern)
        self.log_level_pattern = re.compile(log_level_pattern)
        self.exception_patterns = [re.compile(pattern) for pattern in exception_patterns]
        self.max_stack_depth = max_stack_depth
        self.max_message_length = max_message_length

    def preprocess(self, df: pd.DataFrame, message_column: str = "message") -> pd.DataFrame:
        """
        预处理日志DataFrame

        Args:
            df: 包含日志数据的DataFrame
            message_column: 消息列名

        Returns:
            预处理后的DataFrame
        """
        if df.empty:
            logger.warning("DataFrame为空，无法进行预处理")
            return df

        # 确保消息列存在
        if message_column not in df.columns:
            logger.error(f"消息列 '{message_column}' 不在DataFrame中")
            return df

        # 创建副本以避免修改原始数据
        result_df = df.copy()

        # 处理空值
        result_df[message_column] = result_df[message_column].fillna("").astype(str)

        # 应用文本处理
        logger.info(f"开始预处理 {len(result_df)} 条日志记录")

        # 提取日志级别
        result_df["log_level"] = result_df[message_column].apply(self._extract_log_level)

        # 提取时间戳
        if "timestamp" not in result_df.columns:
            result_df["extracted_timestamp"] = result_df[message_column].apply(self._extract_timestamp)

        # 提取堆栈跟踪
        result_df["stack_trace"] = result_df[message_column].apply(self._extract_stack_trace)

        # 清洗消息内容
        result_df["cleaned_message"] = result_df[message_column].apply(self._clean_message)

        # 截断过长的消息
        result_df["cleaned_message"] = result_df["cleaned_message"].apply(
            lambda x: x[:self.max_message_length] if len(x) > self.max_message_length else x
        )

        logger.info(f"预处理完成，处理了 {len(result_df)} 条日志记录")
        return result_df

    def _extract_log_level(self, text: str) -> str:
        """
        从日志文本中提取日志级别

        Args:
            text: 日志文本

        Returns:
            提取的日志级别，如果没有找到则返回空字符串
        """
        if not text:
            return ""

        # 使用正则表达式匹配日志级别
        match = self.log_level_pattern.search(text)
        if match:
            return match.group(1)

        # 尝试通过关键词判断
        text_lower = text.lower()
        if "error" in text_lower or "exception" in text_lower:
            return "ERROR"
        elif "warn" in text_lower or "warning" in text_lower:
            return "WARN"
        elif "info" in text_lower:
            return "INFO"
        elif "debug" in text_lower:
            return "DEBUG"
        elif "trace" in text_lower:
            return "TRACE"
        elif "fatal" in text_lower:
            return "FATAL"

        # 默认返回INFO
        return "INFO"

    def _extract_timestamp(self, text: str) -> str:
        """
        从日志文本中提取时间戳

        Args:
            text: 日志文本

        Returns:
            提取的时间戳，如果没有找到则返回空字符串
        """
        if not text:
            return ""

        # 使用正则表达式匹配时间戳
        match = self.time_pattern.search(text)
        if match:
            return match.group(0)

        return ""

    def _extract_stack_trace(self, text: str) -> str:
        """
        从日志文本中提取堆栈跟踪

        Args:
            text: 日志文本

        Returns:
            提取的堆栈跟踪，如果没有找到则返回空字符串
        """
        if not text:
            return ""

        # 检查是否包含异常标记
        stack_trace = ""

        for pattern in self.exception_patterns:
            match = pattern.search(text)
            if match:
                # 找到异常标记，提取后续行作为堆栈跟踪
                stack_start = match.start()
                stack_trace = text[stack_start:]

                # 限制堆栈深度
                lines = stack_trace.split("\n")
                if len(lines) > self.max_stack_depth:
                    stack_trace = "\n".join(lines[:self.max_stack_depth])

                break

        return stack_trace

    def _clean_message(self, text: str) -> str:
        """
        清洗日志消息

        Args:
            text: 日志文本

        Returns:
            清洗后的消息
        """
        if not text:
            return ""

        # 移除堆栈跟踪
        cleaned_text = text
        for pattern in self.exception_patterns:
            match = pattern.search(text)
            if match:
                cleaned_text = text[:match.start()].strip()
                break

        # 移除ANSI转义序列
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned_text = ansi_escape.sub('', cleaned_text)

        # 标准化空白字符
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

    def extract_structured_data(self, df: pd.DataFrame, message_column: str = "message") -> pd.DataFrame:
        """
        从日志消息中提取结构化数据

        Args:
            df: 包含日志数据的DataFrame
            message_column: 消息列名

        Returns:
            添加了结构化数据列的DataFrame
        """
        if df.empty:
            return df

        # 创建副本以避免修改原始数据
        result_df = df.copy()

        # 提取JSON数据
        result_df["extracted_json"] = result_df[message_column].apply(self._extract_json)

        # 提取键值对
        result_df["extracted_keyvalue"] = result_df[message_column].apply(self._extract_key_value_pairs)

        # 扁平化提取的结构化数据
        json_df = pd.json_normalize(result_df["extracted_json"])
        kv_df = pd.json_normalize(result_df["extracted_keyvalue"])

        # 为避免列名冲突，添加前缀
        json_df = json_df.add_prefix("json_")
        kv_df = kv_df.add_prefix("kv_")

        # 重置索引以便连接
        json_df.index = result_df.index
        kv_df.index = result_df.index

        # 连接DataFrame
        result_df = pd.concat([result_df, json_df, kv_df], axis=1)

        # 移除中间临时列
        result_df = result_df.drop(columns=["extracted_json", "extracted_keyvalue"], errors="ignore")

        return result_df

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取JSON数据

        Args:
            text: 日志文本

        Returns:
            提取的JSON数据字典
        """
        if not text:
            return {}

        # 查找JSON对象的开始和结束位置
        json_data = {}

        # 查找可能的JSON对象 {..}
        json_regex = r'\{(?:[^{}]|(?R))*\}'
        json_matches = re.finditer(json_regex, text)

        for match in json_matches:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    json_data.update(data)
            except json.JSONDecodeError:
                continue

        return json_data

    def _extract_key_value_pairs(self, text: str) -> Dict[str, str]:
        """
        从文本中提取键值对

        Args:
            text: 日志文本

        Returns:
            提取的键值对字典
        """
        if not text:
            return {}

        # 查找常见的键值对模式
        kv_patterns = [
            r'(\w+)=([^,\s]+)',  # key=value
            r'(\w+):"([^"]*)"',  # key:"value"
            r"(\w+):'([^']*)'",  # key:'value'
            r'(\w+)->([^,\s]+)',  # key->value
            r'(\w+)\s*:\s*([^,\s]+)'  # key: value
        ]

        kv_data = {}

        for pattern in kv_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                key, value = match.groups()
                # 只保留ASCII键名，避免异常字符
                if all(ord(c) < 128 for c in key):
                    kv_data[key] = value

        return kv_data

    def normalize_exceptions(self, df: pd.DataFrame, message_column: str = 'message') -> pd.DataFrame:
        """
        针对各种类型的异常进行标准化处理

        Args:
            df: 日志DataFrame
            message_column: 消息列名

        Returns:
            标准化后的DataFrame
        """
        if df.empty:
            return df

        # 创建副本
        result_df = df.copy()

        # 通用异常模式
        exception_patterns = [
            # ABP框架
            (r'(Volo\.Abp\.[A-Za-z\.]+Exception:.*?)(?=\s+---|\s+at\s+|$)', 'abp_exception'),

            # 通用.NET异常
            (r'(System\.[A-Za-z\.]+Exception:.*?)(?=\s+---|\s+at\s+|$)', 'system_exception'),

            # Java异常
            (r'(java\.[A-Za-z\.]+Exception:.*?)(?=\s+at\s+|$)', 'java_exception'),

            # Python异常
            (r'(Traceback \(most recent call last\):[\s\S]*?(?:Error|Exception):.*?)(?=\n\n|\Z)', 'python_exception'),

            # Node.js/JavaScript异常
            (r'(Error:.*?)(?=\s+at\s+|$)', 'js_exception'),

            # 自定义异常（可以根据需要添加）
            (r'(CustomError:.*?)(?=\n|$)', 'custom_exception')
        ]

        # 提取异常类型和消息
        for pattern, exception_type in exception_patterns:
            # 创建匹配列（用于标记哪些行匹配此模式）
            result_df[f'{exception_type}_match'] = result_df[message_column].str.contains(pattern, regex=True, na=False)

            # 只处理匹配的行
            matched_rows = result_df[result_df[f'{exception_type}_match']]
            if not matched_rows.empty:
                # 提取异常消息
                result_df.loc[result_df[f'{exception_type}_match'], f'{exception_type}_message'] = \
                    matched_rows[message_column].str.extract(pattern, expand=False)

                # 提取异常类名（取第一个冒号前的部分）
                result_df.loc[result_df[f'{exception_type}_match'], f'{exception_type}_class'] = \
                    result_df.loc[result_df[f'{exception_type}_match'], f'{exception_type}_message'].str.split(':',
                                                                                                               n=1).str[
                        0]

            # 删除临时匹配列
            result_df = result_df.drop(columns=[f'{exception_type}_match'])

        # 创建统一的异常字段
        # 从各种特定类型的异常字段中提取信息，合并到通用字段
        result_df['exception_type'] = None
        result_df['exception_class'] = None
        result_df['exception_message'] = None

        # 按优先级填充通用异常字段
        for pattern, exception_type in exception_patterns:
            # 找到有此类异常的行
            if f'{exception_type}_message' in result_df.columns:
                mask = result_df[f'{exception_type}_message'].notna()
                if mask.any():
                    # 填充通用异常字段
                    result_df.loc[mask & result_df['exception_type'].isna(), 'exception_type'] = exception_type
                    result_df.loc[mask & result_df['exception_class'].isna(), 'exception_class'] = \
                        result_df.loc[mask, f'{exception_type}_class']
                    result_df.loc[mask & result_df['exception_message'].isna(), 'exception_message'] = \
                        result_df.loc[mask, f'{exception_type}_message']
            else:
                continue

        return result_df

    def _extract_abp_exception_type(self, text: str) -> str:
        """
        从日志中提取ABP异常类型

        Args:
            text: 日志文本

        Returns:
            异常类型
        """
        # 匹配ABP异常类型
        pattern = r'(Volo\.Abp\.[A-Za-z\.]+Exception)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return ""

    def _extract_abp_module(self, exception_type: str) -> str:
        """
        从异常类型中提取ABP模块名称

        Args:
            exception_type: 异常类型字符串

        Returns:
            模块名称
        """
        if not exception_type:
            return ""

        # 提取模块名称 (Volo.Abp.{Module})
        parts = exception_type.split('.')
        if len(parts) >= 3:
            return parts[2]
        return ""

    def _extract_regex_group(self, text: str, pattern: str, group: int) -> str:
        """
        使用正则表达式提取文本组

        Args:
            text: 源文本
            pattern: 正则表达式模式
            group: 要提取的组索引

        Returns:
            提取的文本，如果没有匹配则返回空字符串
        """
        if not text:
            return ""

        match = re.search(pattern, text)
        if match and group <= len(match.groups()):
            return match.group(group)
        return ""