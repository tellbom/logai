# utils/text_utils.py
import re
import string
import unicodedata
import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union
import jieba
import hashlib
from collections import Counter

logger = logging.getLogger(__name__)

# 常用正则表达式
DATETIME_PATTERN = re.compile(
    r'\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{1,2}(?::\d{1,2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?')
URL_PATTERN = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!./?=&]*)?')
IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
PATH_PATTERN = re.compile(r'(?:/[-\w.]+)+(?:/[-\w.]+)*')
ID_PATTERN = re.compile(r'\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b', re.IGNORECASE)  # UUID
HEXID_PATTERN = re.compile(r'\b[0-9a-f]{24}\b', re.IGNORECASE)  # MongoDB ObjectId等十六进制ID
NUM_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\b')


def clean_text(text: str,
               remove_urls: bool = True,
               remove_paths: bool = True,
               remove_ids: bool = True,
               remove_ips: bool = True,
               remove_dates: bool = True,
               remove_numbers: bool = False,
               normalize_whitespace: bool = True) -> str:
    """
    清洗文本，移除或替换特定模式

    Args:
        text: 输入文本
        remove_urls: 是否移除URL
        remove_paths: 是否移除文件路径
        remove_ids: 是否移除ID（UUID、十六进制ID等）
        remove_ips: 是否移除IP地址
        remove_dates: 是否移除日期时间
        remove_numbers: 是否移除数字
        normalize_whitespace: 是否标准化空白字符

    Returns:
        清洗后的文本
    """
    if not text:
        return ""

    # 创建可能的替换
    if remove_urls:
        text = URL_PATTERN.sub(" [URL] ", text)

    if remove_paths:
        text = PATH_PATTERN.sub(" [PATH] ", text)

    if remove_ids:
        text = ID_PATTERN.sub(" [UUID] ", text)
        text = HEXID_PATTERN.sub(" [ID] ", text)

    if remove_ips:
        text = IP_PATTERN.sub(" [IP] ", text)

    if remove_dates:
        text = DATETIME_PATTERN.sub(" [DATE] ", text)

    if remove_numbers:
        text = NUM_PATTERN.sub(" [NUM] ", text)

    # 标准化空白字符
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_unicode(text: str) -> str:
    """
    Unicode标准化，将特殊字符转换为基本形式

    Args:
        text: 输入文本

    Returns:
        标准化后的文本
    """
    if not text:
        return ""

    # NFKC模式：兼容等价规范化
    return unicodedata.normalize('NFKC', text)


def extract_keywords(text: str,
                     language: str = 'en',
                     max_keywords: int = 10,
                     min_word_length: int = 2) -> List[str]:
    """
    从文本中提取关键词

    Args:
        text: 输入文本
        language: 语言，'en'或'zh'
        max_keywords: 最大关键词数量
        min_word_length: 最小词长度

    Returns:
        关键词列表
    """
    if not text:
        return []

    # 英文关键词提取
    if language.lower() == 'en':
        # 移除标点和转换为小写
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()

        # 分词
        words = text.split()

        # 过滤短词和停用词
        stopwords = get_stopwords(language)
        words = [word for word in words if len(word) >= min_word_length and word not in stopwords]

        # 统计词频
        word_counts = Counter(words)

        # 返回最常见的词
        return [word for word, _ in word_counts.most_common(max_keywords)]

    # 中文关键词提取
    elif language.lower() == 'zh':
        try:
            # 使用jieba分词
            seg_list = jieba.cut(text)

            # 过滤停用词和短词
            stopwords = get_stopwords(language)
            words = [word for word in seg_list if len(word) >= min_word_length and word not in stopwords]

            # 统计词频
            word_counts = Counter(words)

            # 返回最常见的词
            return [word for word, _ in word_counts.most_common(max_keywords)]
        except Exception as e:
            logger.error(f"中文分词出错: {str(e)}")
            return []

    # 未知语言
    else:
        logger.warning(f"不支持的语言: {language}")
        return []


def split_text_chunks(text: str,
                      chunk_size: int = 512,
                      overlap: int = 128,
                      split_by_sentence: bool = True) -> List[str]:
    """
    将长文本分割成较小的块

    Args:
        text: 输入文本
        chunk_size: 块大小（字符数）
        overlap: 重叠大小（字符数）
        split_by_sentence: 是否按句子分割

    Returns:
        文本块列表
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    # 按句子分割
    if split_by_sentence:
        # 将文本分割成句子
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)

        current_chunk = ""

        for sentence in sentences:
            # 如果当前句子本身超过块大小，按字符分割
            if len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # 按字符分割长句子
                for i in range(0, len(sentence), chunk_size - overlap):
                    sub_chunk = sentence[i:i + chunk_size]
                    if sub_chunk:
                        chunks.append(sub_chunk)
            else:
                # 如果添加当前句子会超过块大小，先保存当前块
                if len(current_chunk) + len(sentence) > chunk_size:
                    chunks.append(current_chunk)

                    # 如果启用重叠，新块从上一块的末尾开始
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + sentence
                    else:
                        current_chunk = sentence
                else:
                    # 添加句子到当前块
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)

    # 按字符分割
    else:
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])

    return chunks


def get_stopwords(language: str = 'en') -> Set[str]:
    """
    获取停用词

    Args:
        language: 语言，'en'或'zh'

    Returns:
        停用词集合
    """
    if language.lower() == 'en':
        # 英文常用停用词
        return {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'is', 'in', 'into', 'of', 'for', 'with', 'by', 'to', 'from', 'on',
            'at', 'this', 'that', 'these', 'those', 'it', 'its', 'it\'s', 'i',
            'my', 'me', 'mine', 'we', 'our', 'ours', 'you', 'your', 'yours',
            'he', 'his', 'him', 'she', 'her', 'hers', 'they', 'their', 'them',
            'be', 'been', 'being', 'am', 'are', 'is', 'was', 'were', 'has',
            'have', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
            'can', 'could', 'may', 'might', 'must', 'not', 'no', 'nor', 'so',
        }
    elif language.lower() == 'zh':
        # 中文常用停用词
        return {
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
            '或', '一个', '没有', '我们', '你们', '他们', '她们', '它们',
            '这', '那', '这个', '那个', '这些', '那些', '这样', '那样',
            '之', '得', '地', '在', '上', '下', '前', '后', '里', '外',
            '来', '去', '被', '把', '向', '让', '给', '的话', '说', '对于',
        }
    else:
        logger.warning(f"不支持的语言: {language}")
        return set()


def detect_language(text: str) -> str:
    """
    检测文本语言

    Args:
        text: 输入文本

    Returns:
        检测到的语言代码，'en', 'zh'或'unknown'
    """
    if not text:
        return "unknown"

    # 简单的语言检测逻辑
    # 检查是否包含中文字符
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return "zh"

    # 默认为英文
    return "en"


def calculate_text_hash(text: str) -> str:
    """
    计算文本的哈希值

    Args:
        text: 输入文本

    Returns:
        MD5哈希字符串
    """
    if not text:
        return ""

    return hashlib.md5(text.encode('utf-8')).hexdigest()


def remove_ansi_codes(text: str) -> str:
    """
    移除ANSI转义码（控制台颜色代码等）

    Args:
        text: 输入文本

    Returns:
        清理后的文本
    """
    if not text:
        return ""

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def extract_stacktrace(text: str, max_depth: int = 30) -> str:
    """
    从文本中提取堆栈跟踪

    Args:
        text: 输入文本
        max_depth: 最大堆栈深度

    Returns:
        提取的堆栈跟踪
    """
    if not text:
        return ""

    # 常见异常标记
    exception_patterns = [
        r'(Exception:.*?)(?=\s+at\s+|\s+---|\Z)',
        r'(Error:.*?)(?=\s+at\s+|\s+---|\Z)',
        r'(Traceback \(most recent call last\):.*?)(?=\Z)',
        r'(错误:.*?)(?=\s+在\s+|\Z)',
        r'(异常:.*?)(?=\s+在\s+|\Z)',
    ]

    # 尝试各种模式
    for pattern in exception_patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            stack_trace = matches.group(1).strip()

            # 限制堆栈行数
            lines = stack_trace.splitlines()
            if len(lines) > max_depth:
                stack_trace = '\n'.join(lines[:max_depth]) + "\n... (stack trace truncated)"

            return stack_trace

    return ""


def find_similar_texts(texts: List[str], threshold: float = 0.8) -> List[Tuple[int, int, float]]:
    """
    查找相似文本对

    Args:
        texts: 文本列表
        threshold: 相似度阈值

    Returns:
        相似文本对列表，每项为(索引1, 索引2, 相似度)
    """
    from difflib import SequenceMatcher

    if not texts or len(texts) < 2:
        return []

    similar_pairs = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # 计算两个文本的相似度
            similarity = SequenceMatcher(None, texts[i], texts[j]).ratio()

            if similarity >= threshold:
                similar_pairs.append((i, j, similarity))

    # 按相似度降序排序
    return sorted(similar_pairs, key=lambda x: x[2], reverse=True)


def is_json_string(text: str) -> bool:
    """
    检查字符串是否为有效的JSON

    Args:
        text: 输入文本

    Returns:
        是否为有效JSON
    """
    if not text:
        return False

    import json

    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def extract_key_value_pairs(text: str) -> Dict[str, str]:
    """
    从文本中提取键值对

    Args:
        text: 输入文本

    Returns:
        键值对字典
    """
    if not text:
        return {}

    result = {}

    # 常见的键值对模式
    patterns = [
        r'(\w+)=([^,\s]+)',  # key=value
        r'(\w+):"([^"]*)"',  # key:"value"
        r"(\w+):'([^']*)'",  # key:'value'
        r'(\w+)->([^,\s]+)',  # key->value
        r'(\w+)\s*:\s*([^,\s]+)'  # key: value
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            key, value = match.groups()
            # 只保留ASCII键名，避免异常字符
            if all(ord(c) < 128 for c in key):
                result[key] = value

    return result


def format_exception(exc: Exception) -> str:
    """
    格式化异常信息为可读字符串

    Args:
        exc: 异常对象

    Returns:
        格式化的异常信息
    """
    import traceback

    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return ''.join(tb_lines)