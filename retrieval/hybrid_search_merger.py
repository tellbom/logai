import re
import math
import jieba
import jieba.analyse
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class HybridSearchMerger:
    """混合搜索结果合并器，优化中英文混合检索"""

    def __init__(self):
        # 初始化jieba
        self._init_jieba()

        # 中文停用词
        self.chinese_stopwords = {
            '的', '了', '和', '是', '在', '有', '我', '你', '他', '这', '那', '之', '与', '及', '其',
            '到', '以', '会', '也', '但', '对', '从', '而', '被', '把', '给', '让', '向', '往', '为',
            '所', '于', '就', '不', '都', '已', '可', '里', '去', '过', '她', '它', '个', '们', '来'
        }

        # 英文停用词
        self.english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these'
        }

    def _init_jieba(self):
        """初始化jieba分词器，添加技术术语"""
        # 添加技术相关词汇
        tech_words = [
            'API', 'HTTP', 'HTTPS', 'REST', 'JSON', 'XML', 'SQL', 'NoSQL',
            'TCP', 'UDP', 'SSL', 'TLS', 'DNS', 'CDN', 'OAuth', 'JWT',
            'Exception', 'Error', 'Warning', 'Debug', 'Trace', 'Fatal',
            'timeout', 'connection', 'database', 'server', 'client',
            'authentication', 'authorization', 'permission', 'token',
            '异常', '错误', '警告', '调试', '跟踪', '致命错误',
            '超时', '连接', '数据库', '服务器', '客户端', '认证', '授权'
        ]

        for word in tech_words:
            jieba.add_word(word)

    def _merge_final_results(
            self,
            query: str,
            es_results: List[Dict],
            vector_results: List[Dict],
            reranked_results: List[Dict],
            es_weight: float = 0.3,
            vector_weight: float = 0.5,
            rerank_weight: float = 0.2,
            final_results_limit: int = 20
    ) -> List[Dict]:
        """
        优化的结果合并算法 - 针对中英文混合日志

        Args:
            query: 搜索查询
            es_results: ES搜索结果
            vector_results: 向量搜索结果
            reranked_results: 重排序结果
            es_weight: ES权重
            vector_weight: 向量权重
            rerank_weight: 重排序权重
            final_results_limit: 返回结果数限制

        Returns:
            合并后的结果列表
        """
        # 创建基础合并结果
        result_map = self._create_base_result_map(es_results, vector_results, reranked_results)
        combined_results = list(result_map.values())

        if not combined_results:
            return []

        # 分析查询特征
        query_features = self._analyze_query_features_enhanced(query)
        logger.info(f"查询特征分析: {query_features}")

        # 获取自适应权重
        adaptive_es_weight, adaptive_vector_weight, adaptive_rerank_weight = self._get_adaptive_weights_enhanced(
            query, query_features, es_results, vector_results, es_weight, vector_weight, rerank_weight
        )

        # 计算各种相关性分数
        chinese_relevance = self._calculate_chinese_relevance(query, combined_results, query_features)
        english_relevance = self._calculate_english_relevance(query, combined_results, query_features)
        mixed_relevance = self._calculate_mixed_language_relevance(query, combined_results, query_features)
        tfidf_scores = self._calculate_tfidf_relevance_enhanced(query, combined_results)
        context_scores = self._calculate_context_score(combined_results)
        keyword_scores = self._calculate_keyword_relevance_enhanced(query, combined_results, query_features)

        # 处理每个结果，计算综合分数
        for result in combined_results:
            # 基础分数
            base_score = (
                    adaptive_es_weight * result.get("es_score", 0.0) +
                    adaptive_vector_weight * result.get("vector_score", 0.0) +
                    adaptive_rerank_weight * result.get("rerank_score", 0.0)
            )

            # 获取各项评分
            level_importance = self._get_level_importance(result.get("log_level", "INFO"))
            tfidf_score = tfidf_scores.get(result["id"], 0.0)
            context_score = context_scores.get(result["id"], 1.0)
            keyword_score = keyword_scores.get(result["id"], 0.0)

            # 语言相关性分数（根据查询特征选择）
            if query_features["is_mixed_language"]:
                language_score = mixed_relevance.get(result["id"], 0.0)
            elif query_features["is_primarily_chinese"]:
                language_score = chinese_relevance.get(result["id"], 0.0)
            else:
                language_score = english_relevance.get(result["id"], 0.0)

            # 查询-文档匹配度
            query_doc_match = self._calculate_query_document_match_enhanced(query, result, query_features)

            # 长度惩罚（考虑中英文差异）
            length_penalty = self._calculate_length_penalty_enhanced(result.get("message", ""))

            # 时间新鲜度
            time_freshness = self._calculate_time_freshness(result.get("timestamp", ""))

            # 动态计算最终分数权重
            score_weights = self._get_dynamic_score_weights(query_features)

            # 综合分数计算
            final_score = (
                                  base_score * score_weights["base"] +
                                  language_score * score_weights["language"] +
                                  keyword_score * score_weights["keyword"] +
                                  query_doc_match * score_weights["match"] +
                                  tfidf_score * score_weights["tfidf"] +
                                  (level_importance * score_weights["level_weight"] +
                                   context_score * score_weights["context_weight"] +
                                   time_freshness * score_weights["time_weight"]) * score_weights["other"]
                          ) * length_penalty

            result["final_score"] = final_score

            # 详细分数分解（用于调试）
            result["score_breakdown"] = {
                "base_score": base_score,
                "language_score": language_score,
                "keyword_score": keyword_score,
                "query_doc_match": query_doc_match,
                "tfidf_score": tfidf_score,
                "level_importance": level_importance,
                "context_score": context_score,
                "time_freshness": time_freshness,
                "length_penalty": length_penalty,
                "weights_used": score_weights
            }

        # 排序和限制结果
        combined_results.sort(key=lambda x: x["final_score"], reverse=True)
        combined_results = combined_results[:final_results_limit]

        # 添加排名和相关性信息
        for i, result in enumerate(combined_results):
            result["rank"] = i + 1
            self._add_relevance_info_enhanced(query, result, query_features)

        return combined_results

    def _analyze_query_features_enhanced(self, query: str) -> Dict[str, Any]:
        """增强的查询特征分析，使用jieba分词"""
        # 基础特征
        features = {
            "original_query": query,
            "has_chinese": bool(re.search(r'[\u4e00-\u9fff]', query)),
            "has_english": bool(re.search(r'[a-zA-Z]', query)),
            "has_numbers": bool(re.search(r'\d', query)),
            "has_special_chars": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', query)),
            "query_length": len(query),
            "is_mixed_language": False,
            "is_primarily_chinese": False,
            "is_primarily_english": False,
            "query_type": "general",
        }

        # 计算语言比例
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', query)
        english_chars = re.findall(r'[a-zA-Z]', query)
        total_chars = len(query.replace(" ", ""))

        if total_chars > 0:
            features["chinese_ratio"] = len(chinese_chars) / total_chars
            features["english_ratio"] = len(english_chars) / total_chars
        else:
            features["chinese_ratio"] = 0
            features["english_ratio"] = 0

        # 判断语言类型
        features["is_mixed_language"] = features["has_chinese"] and features["has_english"]
        features["is_primarily_chinese"] = features["chinese_ratio"] > 0.6
        features["is_primarily_english"] = features["english_ratio"] > 0.6

        # 中文分词
        if features["has_chinese"]:
            chinese_text = ''.join(chinese_chars)
            # 使用jieba进行分词
            chinese_words = list(jieba.cut(chinese_text, cut_all=False))
            # 过滤停用词和单字
            features["chinese_keywords"] = [
                w for w in chinese_words
                if len(w) > 1 and w not in self.chinese_stopwords
            ]
            # 提取关键词（TF-IDF）
            features["chinese_key_terms"] = jieba.analyse.extract_tags(
                chinese_text, topK=5, withWeight=False
            )
        else:
            features["chinese_keywords"] = []
            features["chinese_key_terms"] = []

        # 英文分词
        if features["has_english"]:
            english_words = re.findall(r'[a-zA-Z]+', query.lower())
            features["english_keywords"] = [
                w for w in english_words
                if len(w) > 2 and w not in self.english_stopwords
            ]
        else:
            features["english_keywords"] = []

        # 提取数字和特殊模式
        features["numbers"] = re.findall(r'\d+', query)
        features["ip_addresses"] = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', query)
        features["urls"] = re.findall(r'https?://[^\s]+', query)

        # 检测技术术语
        tech_patterns = {
            'error_terms': r'(?i)(error|exception|fail|crash|错误|异常|失败|崩溃)',
            'api_terms': r'(?i)(api|rest|http|request|response|请求|响应|接口)',
            'auth_terms': r'(?i)(auth|login|token|permission|认证|登录|权限|令牌)',
            'db_terms': r'(?i)(database|sql|query|transaction|数据库|查询|事务)',
            'timeout_terms': r'(?i)(timeout|slow|delay|超时|缓慢|延迟)'
        }

        features["technical_terms"] = {}
        for term_type, pattern in tech_patterns.items():
            matches = re.findall(pattern, query)
            if matches:
                features["technical_terms"][term_type] = matches

        # 确定查询类型
        if features["technical_terms"].get("error_terms"):
            features["query_type"] = "error_investigation"
        elif features["technical_terms"].get("timeout_terms"):
            features["query_type"] = "performance_analysis"
        elif features["technical_terms"].get("auth_terms"):
            features["query_type"] = "security_audit"
        elif features["is_mixed_language"]:
            features["query_type"] = "mixed_language_query"
        elif len(query.split()) <= 2:
            features["query_type"] = "keyword_search"
        elif '?' in query or any(q in query for q in ['什么', '如何', '为什么', 'what', 'how', 'why']):
            features["query_type"] = "question_query"

        return features

    def _calculate_chinese_relevance(
            self,
            query: str,
            results: List[Dict],
            query_features: Dict
    ) -> Dict[str, float]:
        """计算中文相关性分数"""
        relevance_scores = {}

        # 获取查询的中文关键词
        query_keywords = set(query_features.get("chinese_keywords", []))
        query_key_terms = set(query_features.get("chinese_key_terms", []))

        if not query_keywords and not query_key_terms:
            return {result["id"]: 0.0 for result in results}

        for result in results:
            message = result.get("message", "")

            # 提取消息中的中文文本
            chinese_text = ''.join(re.findall(r'[\u4e00-\u9fff]+', message))
            if not chinese_text:
                relevance_scores[result["id"]] = 0.0
                continue

            # 对消息进行分词
            message_words = set(jieba.cut(chinese_text, cut_all=False))
            message_words = {w for w in message_words if len(w) > 1 and w not in self.chinese_stopwords}

            # 计算关键词重叠度
            keyword_overlap = len(query_keywords & message_words) / len(query_keywords) if query_keywords else 0

            # 计算关键术语匹配度
            key_term_matches = sum(1 for term in query_key_terms if term in chinese_text)
            key_term_score = key_term_matches / len(query_key_terms) if query_key_terms else 0

            # 使用jieba的TF-IDF提取消息关键词并计算相似度
            message_key_terms = set(jieba.analyse.extract_tags(chinese_text, topK=10, withWeight=False))
            tfidf_overlap = len(query_key_terms & message_key_terms) / len(
                query_key_terms | message_key_terms
            ) if (query_key_terms | message_key_terms) else 0

            # 综合相关性分数
            relevance_score = (
                    keyword_overlap * 0.4 +
                    key_term_score * 0.4 +
                    tfidf_overlap * 0.2
            )

            relevance_scores[result["id"]] = min(1.0, relevance_score)

        return relevance_scores

    def _calculate_english_relevance(
            self,
            query: str,
            results: List[Dict],
            query_features: Dict
    ) -> Dict[str, float]:
        """计算英文相关性分数"""
        relevance_scores = {}

        # 获取查询的英文关键词
        query_keywords = set(query_features.get("english_keywords", []))

        if not query_keywords:
            return {result["id"]: 0.0 for result in results}

        for result in results:
            message = result.get("message", "").lower()

            # 提取消息中的英文单词
            message_words = set(re.findall(r'[a-zA-Z]+', message))
            message_words = {w for w in message_words if len(w) > 2 and w not in self.english_stopwords}

            if not message_words:
                relevance_scores[result["id"]] = 0.0
                continue

            # 计算精确匹配
            exact_matches = len(query_keywords & message_words)
            exact_score = exact_matches / len(query_keywords)

            # 计算部分匹配（词干相似）
            partial_matches = 0
            for q_word in query_keywords:
                for m_word in message_words:
                    if (q_word in m_word or m_word in q_word) and q_word != m_word:
                        partial_matches += 0.5
                        break

            partial_score = partial_matches / len(query_keywords) if query_keywords else 0

            # 综合相关性分数
            relevance_score = exact_score * 0.7 + partial_score * 0.3
            relevance_scores[result["id"]] = min(1.0, relevance_score)

        return relevance_scores

    def _calculate_mixed_language_relevance(
            self,
            query: str,
            results: List[Dict],
            query_features: Dict
    ) -> Dict[str, float]:
        """计算中英文混合相关性分数"""
        # 分别计算中文和英文相关性
        chinese_relevance = self._calculate_chinese_relevance(query, results, query_features)
        english_relevance = self._calculate_english_relevance(query, results, query_features)

        # 根据查询中的语言比例加权合并
        mixed_relevance = {}
        chinese_ratio = query_features.get("chinese_ratio", 0.5)
        english_ratio = query_features.get("english_ratio", 0.5)

        # 归一化比例
        total_ratio = chinese_ratio + english_ratio
        if total_ratio > 0:
            chinese_weight = chinese_ratio / total_ratio
            english_weight = english_ratio / total_ratio
        else:
            chinese_weight = english_weight = 0.5

        for result in results:
            result_id = result["id"]
            cn_score = chinese_relevance.get(result_id, 0.0)
            en_score = english_relevance.get(result_id, 0.0)

            # 加权合并，但也考虑两种语言都匹配的情况给予奖励
            base_score = cn_score * chinese_weight + en_score * english_weight

            # 如果两种语言都有匹配，给予额外奖励
            if cn_score > 0.3 and en_score > 0.3:
                bonus = min(cn_score, en_score) * 0.2
                mixed_score = min(1.0, base_score + bonus)
            else:
                mixed_score = base_score

            mixed_relevance[result_id] = mixed_score

        return mixed_relevance

    def _calculate_keyword_relevance_enhanced(
            self,
            query: str,
            results: List[Dict],
            query_features: Dict
    ) -> Dict[str, float]:
        """增强的关键词相关性计算"""
        keyword_scores = {}

        # 收集所有关键词及其权重
        weighted_keywords = []

        # 中文关键词（权重更高）
        for keyword in query_features.get("chinese_keywords", []):
            weighted_keywords.append((keyword, 1.5))

        # 中文关键术语（权重最高）
        for keyword in query_features.get("chinese_key_terms", []):
            weighted_keywords.append((keyword, 2.0))

        # 英文关键词
        for keyword in query_features.get("english_keywords", []):
            weighted_keywords.append((keyword, 1.0))

        # 技术术语（根据类型给不同权重）
        tech_weights = {
            'error_terms': 2.5,
            'api_terms': 1.8,
            'auth_terms': 1.8,
            'db_terms': 1.5,
            'timeout_terms': 2.0
        }

        for term_type, terms in query_features.get("technical_terms", {}).items():
            weight = tech_weights.get(term_type, 1.5)
            for term in terms:
                weighted_keywords.append((term.lower(), weight))

        # 数字和特殊模式
        for number in query_features.get("numbers", []):
            weighted_keywords.append((number, 1.2))

        for ip in query_features.get("ip_addresses", []):
            weighted_keywords.append((ip, 2.0))

        if not weighted_keywords:
            return {result["id"]: 0.0 for result in results}

        # 计算每个结果的关键词得分
        for result in results:
            message = result.get("message", "")
            message_lower = message.lower()

            total_weight = sum(w for _, w in weighted_keywords)
            matched_weight = 0

            for keyword, weight in weighted_keywords:
                # 精确匹配
                if keyword in message or keyword in message_lower:
                    matched_weight += weight
                # 部分匹配（稍微降权）
                elif len(keyword) > 3 and any(keyword in word for word in message_lower.split()):
                    matched_weight += weight * 0.7

            keyword_scores[result["id"]] = matched_weight / total_weight if total_weight > 0 else 0

        return keyword_scores

    def _calculate_tfidf_relevance_enhanced(
            self,
            query: str,
            results: List[Dict]
    ) -> Dict[str, float]:
        """增强的TF-IDF相关性计算，支持中英文"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not results:
            return {}

        # 自定义分词器
        def mixed_tokenizer(text):
            tokens = []

            # 中文分词
            chinese_text = ''.join(re.findall(r'[\u4e00-\u9fff]+', text))
            if chinese_text:
                chinese_tokens = jieba.cut(chinese_text, cut_all=False)
                tokens.extend([t for t in chinese_tokens if len(t) > 1 and t not in self.chinese_stopwords])

            # 英文分词
            english_tokens = re.findall(r'[a-zA-Z]+', text.lower())
            tokens.extend([t for t in english_tokens if len(t) > 2 and t not in self.english_stopwords])

            # 数字
            tokens.extend(re.findall(r'\d+', text))

            return tokens

        try:
            # 准备文档
            docs = [result.get("message", "") for result in results]
            all_docs = [query] + docs

            # TF-IDF向量化
            vectorizer = TfidfVectorizer(
                tokenizer=mixed_tokenizer,
                token_pattern=None,
                lowercase=False,  # 已在tokenizer中处理
                max_features=1000
            )

            tfidf_matrix = vectorizer.fit_transform(all_docs)

            # 计算相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            # 创建得分映射
            tfidf_scores = {}
            for i, result in enumerate(results):
                tfidf_scores[result["id"]] = float(similarities[i])

            return tfidf_scores

        except Exception as e:
            logger.warning(f"TF-IDF计算失败: {str(e)}")
            return {result["id"]: 0.0 for result in results}

    def _calculate_query_document_match_enhanced(
            self,
            query: str,
            result: Dict,
            query_features: Dict
    ) -> float:
        """增强的查询-文档匹配度计算"""
        message = result.get("message", "")

        if not message:
            return 0.0

        # 1. 长度相似度
        query_len = len(query)
        message_len = len(message)
        length_ratio = min(query_len, message_len) / max(query_len, message_len, 1)

        # 2. 语言类型匹配度
        message_has_chinese = bool(re.search(r'[\u4e00-\u9fff]', message))
        message_has_english = bool(re.search(r'[a-zA-Z]', message))

        language_match = 1.0
        if query_features["has_chinese"] and not message_has_chinese:
            language_match *= 0.7
        if query_features["has_english"] and not message_has_english:
            language_match *= 0.7
        if query_features["is_mixed_language"]:
            if message_has_chinese and message_has_english:
                language_match *= 1.2  # 奖励同样是混合语言的文档

        # 3. 查询类型与日志级别的匹配度
        log_level = result.get("log_level", "").upper()
        query_type = query_features.get("query_type", "general")

        type_match = 1.0
        if query_type == "error_investigation":
            if log_level in ["ERROR", "FATAL", "EXCEPTION"]:
                type_match = 1.5
            elif log_level in ["WARN", "WARNING"]:
                type_match = 1.2
        elif query_type == "performance_analysis":
            if "timeout" in message.lower() or "slow" in message.lower():
                type_match = 1.3
        elif query_type == "security_audit":
            if log_level in ["WARN", "WARNING", "ERROR"] and any(
                    term in message.lower() for term in ['auth', 'permission', 'denied', 'forbidden']
            ):
                type_match = 1.4

        # 4. 序列相似度（考虑词序）
        seq_similarity = SequenceMatcher(None, query.lower(), message.lower()).ratio()

        # 综合匹配度
        match_score = (
                length_ratio * 0.2 +
                language_match * 0.3 +
                type_match * 0.3 +
                seq_similarity * 0.2
        )

        return min(1.0, match_score)

    def _calculate_length_penalty_enhanced(self, message: str) -> float:
        """增强的长度惩罚计算，考虑中英文字符差异"""
        if not message:
            return 0.5

        # 计算有效长度
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', message))
        english_chars = len(re.findall(r'[a-zA-Z]', message))
        numbers = len(re.findall(r'\d', message))
        other_chars = len(message) - chinese_chars - english_chars - numbers - message.count(' ')

        # 中文字符算1.5个单位，英文和数字算1个单位
        effective_length = chinese_chars * 1.5 + english_chars + numbers + other_chars * 0.5

        # 理想长度范围 100-800 有效单位
        if 100 <= effective_length <= 800:
            return 1.0
        elif effective_length < 100:
            # 过短的日志轻微惩罚
            return 0.8 + (effective_length / 100) * 0.2
        elif effective_length <= 1500:
            # 稍长的日志轻微惩罚
            return 0.9
        else:
            # 过长的日志按比例惩罚
            penalty = max(0.6, 1.0 - (effective_length - 1500) / 3000)
            return penalty

    def _get_dynamic_score_weights(self, query_features: Dict) -> Dict[str, float]:
        """根据查询特征动态调整评分权重"""
        query_type = query_features.get("query_type", "general")

        # 基础权重配置
        weight_configs = {
            "error_investigation": {
                "base": 0.30,
                "language": 0.20,
                "keyword": 0.30,
                "match": 0.10,
                "tfidf": 0.05,
                "other": 0.05,
                "level_weight": 0.6,
                "context_weight": 0.3,
                "time_weight": 0.1
            },
            "performance_analysis": {
                "base": 0.25,
                "language": 0.25,
                "keyword": 0.25,
                "match": 0.15,
                "tfidf": 0.05,
                "other": 0.05,
                "level_weight": 0.4,
                "context_weight": 0.4,
                "time_weight": 0.2
            },
            "security_audit": {
                "base": 0.35,
                "language": 0.20,
                "keyword": 0.25,
                "match": 0.10,
                "tfidf": 0.05,
                "other": 0.05,
                "level_weight": 0.5,
                "context_weight": 0.3,
                "time_weight": 0.2
            },
            "mixed_language_query": {
                "base": 0.30,
                "language": 0.35,
                "keyword": 0.20,
                "match": 0.10,
                "tfidf": 0.05,
                "other": 0.00,
                "level_weight": 0.4,
                "context_weight": 0.4,
                "time_weight": 0.2
            },
            "keyword_search": {
                "base": 0.35,
                "language": 0.15,
                "keyword": 0.35,
                "match": 0.10,
                "tfidf": 0.05,
                "other": 0.00,
                "level_weight": 0.4,
                "context_weight": 0.4,
                "time_weight": 0.2
            },
            "question_query": {
                "base": 0.25,
                "language": 0.40,
                "keyword": 0.15,
                "match": 0.10,
                "tfidf": 0.10,
                "other": 0.00,
                "level_weight": 0.3,
                "context_weight": 0.5,
                "time_weight": 0.2
            },
            "general": {
                "base": 0.30,
                "language": 0.30,
                "keyword": 0.20,
                "match": 0.10,
                "tfidf": 0.05,
                "other": 0.05,
                "level_weight": 0.4,
                "context_weight": 0.4,
                "time_weight": 0.2
            }
        }

        return weight_configs.get(query_type, weight_configs["general"])

    def _get_adaptive_weights_enhanced(
            self,
            query: str,
            query_features: Dict,
            es_results: List[Dict],
            vector_results: List[Dict],
            base_es_weight: float,
            base_vector_weight: float,
            base_rerank_weight: float
    ) -> Tuple[float, float, float]:
        """增强的自适应权重调整"""
        query_type = query_features.get("query_type", "general")

        # 权重调整策略
        if query_type == "error_investigation":
            # 错误调查更依赖关键词匹配
            es_weight = 0.5
            vector_weight = 0.35
            rerank_weight = 0.15
        elif query_type == "mixed_language_query":
            # 中英混合查询需要平衡
            es_weight = 0.4
            vector_weight = 0.45
            rerank_weight = 0.15
        elif query_type == "keyword_search":
            # 短关键词查询依赖ES
            es_weight = 0.6
            vector_weight = 0.25
            rerank_weight = 0.15
        elif query_type == "question_query":
            # 问题型查询依赖语义理解
            es_weight = 0.25
            vector_weight = 0.6
            rerank_weight = 0.15
        else:
            # 使用基础权重
            es_weight = base_es_weight
            vector_weight = base_vector_weight
            rerank_weight = base_rerank_weight

        # 根据结果质量微调
        if es_results and vector_results:
            es_avg_score = sum(r.get("score", 0) for r in es_results[:5]) / min(5, len(es_results))
            vector_avg_score = sum(r.get("score", 0) for r in vector_results[:5]) / min(5, len(vector_results))

            # 如果某种搜索的结果明显更好，增加其权重
            if es_avg_score > vector_avg_score * 1.5:
                es_weight = min(0.7, es_weight + 0.1)
                vector_weight = max(0.2, vector_weight - 0.1)
            elif vector_avg_score > es_avg_score * 1.5:
                vector_weight = min(0.7, vector_weight + 0.1)
                es_weight = max(0.2, es_weight - 0.1)

        # 归一化权重
        total_weight = es_weight + vector_weight + rerank_weight
        if total_weight > 0:
            es_weight /= total_weight
            vector_weight /= total_weight
            rerank_weight /= total_weight

        logger.info(
            f"自适应权重调整 - 查询类型: {query_type}, ES: {es_weight:.2f}, Vector: {vector_weight:.2f}, Rerank: {rerank_weight:.2f}")

        return es_weight, vector_weight, rerank_weight

    def _calculate_time_freshness(self, timestamp: str) -> float:
        """计算时间新鲜度分数"""
        if not timestamp:
            return 0.9

        try:
            # 解析时间戳
            if isinstance(timestamp, str):
                timestamp = timestamp.replace("Z", "+00:00")
                log_time = datetime.fromisoformat(timestamp)
            else:
                log_time = timestamp

            # 计算时间差
            now = datetime.now(log_time.tzinfo) if log_time.tzinfo else datetime.now()
            time_diff = (now - log_time).total_seconds()

            # 时间新鲜度评分
            if time_diff < 3600:  # 1小时内
                return 1.2
            elif time_diff < 86400:  # 24小时内
                return 1.1
            elif time_diff < 604800:  # 7天内
                return 1.0
            elif time_diff < 2592000:  # 30天内
                return 0.95
            else:
                return 0.9

        except Exception as e:
            logger.warning(f"时间解析失败: {str(e)}")
            return 0.9

    def _calculate_context_score(self, results: List[Dict]) -> Dict[str, float]:
        """计算上下文相关性分数"""
        context_scores = {result["id"]: 1.0 for result in results}

        if len(results) < 2:
            return context_scores

        # 提取有效时间戳
        timestamped_results = []
        for result in results:
            timestamp_str = result.get("timestamp")
            if timestamp_str:
                try:
                    timestamp_str = timestamp_str.replace("Z", "+00:00")
                    timestamp = datetime.fromisoformat(timestamp_str)
                    timestamped_results.append((result, timestamp))
                except:
                    continue

        # 按时间排序
        timestamped_results.sort(key=lambda x: x[1])

        # 分析时间相近的日志
        for i in range(len(timestamped_results) - 1):
            curr_result, curr_time = timestamped_results[i]
            next_result, next_time = timestamped_results[i + 1]

            time_diff = (next_time - curr_time).total_seconds()

            # 时间接近的日志可能相关
            if time_diff < 300:  # 5分钟内
                boost = 1.0 + (1.0 - time_diff / 300) * 0.2
                context_scores[curr_result["id"]] *= boost
                context_scores[next_result["id"]] *= boost

            # 检查服务相关性
            if curr_result.get("service_name") == next_result.get("service_name"):
                context_scores[curr_result["id"]] *= 1.05
                context_scores[next_result["id"]] *= 1.05

        # 归一化分数
        max_score = max(context_scores.values()) if context_scores else 1.0
        if max_score > 1.0:
            for result_id in context_scores:
                context_scores[result_id] /= max_score

        return context_scores

    def _get_level_importance(self, log_level: str) -> float:
        """获取日志级别的重要性权重"""
        level_weights = {
            "FATAL": 1.0,
            "CRITICAL": 1.0,
            "ERROR": 0.9,
            "EXCEPTION": 0.9,
            "WARN": 0.7,
            "WARNING": 0.7,
            "INFO": 0.5,
            "DEBUG": 0.3,
            "TRACE": 0.2
        }
        return level_weights.get(log_level.upper(), 0.5)

    def _create_base_result_map(
            self,
            es_results: List[Dict[str, Any]],
            vector_results: List[Dict[str, Any]],
            reranked_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """创建基础结果映射"""
        result_map = {}

        # 处理ES结果
        for result in es_results:
            result_id = result["id"]
            result_map[result_id] = {
                **result,
                "es_score": result.get("score", 0.0),
                "vector_score": 0.0,
                "rerank_score": 0.0
            }

        # 处理向量结果
        for result in vector_results:
            result_id = result["id"]
            if result_id in result_map:
                result_map[result_id]["vector_score"] = result.get("score", 0.0)
            else:
                result_map[result_id] = {
                    **result,
                    "es_score": 0.0,
                    "vector_score": result.get("score", 0.0),
                    "rerank_score": 0.0
                }

        # 处理重排序结果
        for result in reranked_results:
            result_id = result["id"]
            if result_id in result_map:
                result_map[result_id]["rerank_score"] = result.get("rerank_score", 0.0)

        return result_map

    def _add_relevance_info_enhanced(self, query: str, result: Dict, query_features: Dict) -> None:
        """为单个结果添加增强的相关性信息"""
        message = result.get("message", "")

        # 收集所有关键词
        all_keywords = (
                query_features.get("chinese_keywords", []) +
                query_features.get("english_keywords", []) +
                query_features.get("numbers", [])
        )

        # 计算关键词覆盖率
        matched_keywords = []
        for keyword in all_keywords:
            if keyword in message or keyword.lower() in message.lower():
                matched_keywords.append(keyword)

        keyword_coverage = len(matched_keywords) / len(all_keywords) if all_keywords else 0

        # 相关性级别
        final_score = result.get("final_score", 0)
        if final_score > 0.8:
            relevance_level = "high"
        elif final_score > 0.5:
            relevance_level = "medium"
        else:
            relevance_level = "low"

        # 添加相关性详情
        result["relevance_info"] = {
            "level": relevance_level,
            "keyword_coverage": keyword_coverage,
            "matched_keywords": matched_keywords,
            "total_keywords": len(all_keywords),
            "query_type": query_features.get("query_type", "general"),
            "language_match": {
                "chinese": query_features["has_chinese"] and bool(re.search(r'[\u4e00-\u9fff]', message)),
                "english": query_features["has_english"] and bool(re.search(r'[a-zA-Z]', message)),
                "mixed": query_features["is_mixed_language"]
            }
        }