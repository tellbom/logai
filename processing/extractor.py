# processing/extractor.py
import logging
import json
import os
import re
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler

from data.es_connector import ESConnector
from utils import get_logger, clean_text

from config import get_config, get_model_config

logger = get_logger(__name__)


class LogProcessor:
    """日志处理器，负责日志解析、异常检测和聚类"""

    def __init__(
            self,
            es_connector: ESConnector,
            target_es_connector: ESConnector,
            output_dir: str = "./output",
            create_output_dir: bool = True
    ):
        """
        初始化日志处理器

        Args:
            es_connector: ES连接器
            output_dir: 输出目录
            create_output_dir: 是否创建输出目录
        """
        self.es_connector = es_connector
        self.target_es_connector = target_es_connector
        self.output_dir = output_dir

        # 创建输出目录
        if create_output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建输出目录: {output_dir}")

        # 状态跟踪
        self.last_processed_timestamp = None

        # 加载和保存最近处理的时间戳
        self.timestamp_file = os.path.join(self.output_dir, "last_timestamp.txt")
        self.load_last_timestamp()

        # 初始化模板存储
        self.template_store = {}
        self.cluster_store = {}

        logger.info("日志处理器初始化完成")

    def save_last_timestamp(self):
        """保存最后处理的时间戳到文件"""
        if self.last_processed_timestamp:
            try:
                with open(self.timestamp_file, "w") as f:
                    f.write(self.last_processed_timestamp)
                logger.info(f"已保存最后处理时间戳: {self.last_processed_timestamp}")
            except Exception as e:
                logger.error(f"保存时间戳失败: {str(e)}")

    def load_last_timestamp(self):
        """从文件加载最后处理的时间戳"""
        if os.path.exists(self.timestamp_file):
            try:
                with open(self.timestamp_file, "r") as f:
                    self.last_processed_timestamp = f.read().strip()
                logger.info(f"已加载最后处理时间戳: {self.last_processed_timestamp}")
            except Exception as e:
                logger.error(f"加载时间戳失败: {str(e)}")

    def parse_logs(
            self,
            logs_df: pd.DataFrame,
            content_column: str = "message",
            timestamp_column: str = "@timestamp",
            id_column: str = "_id",
            algorithm: str = "drain",
            algorithm_params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        解析日志内容，提取模板

        Args:
            logs_df: 包含日志数据的DataFrame
            content_column: 内容列名
            timestamp_column: 时间戳列名
            id_column: ID列名
            algorithm: 解析算法，默认为'drain'
            algorithm_params: 算法参数

        Returns:
            解析后的DataFrame和解析结果
        """
        if logs_df.empty:
            logger.warning("没有日志数据需要解析")
            return logs_df, {"status": "no_data"}

        logger.info(f"开始解析 {len(logs_df)} 条日志记录")

        # 确保内容列存在
        if content_column not in logs_df.columns:
            logger.error(f"内容列 '{content_column}' 不在DataFrame中")
            return logs_df, {"status": "error", "message": f"找不到内容列 '{content_column}'"}

        # 设置默认参数
        if algorithm_params is None:
            algorithm_params = {
                "sim_th": 0.4,
                "depth": 4
            }

        # 创建结果DataFrame的副本
        result_df = logs_df.copy()

        try:
            # 根据算法选择解析方法
            if algorithm.lower() == "drain":
                result_df, templates = self._drain_parse(result_df, content_column, algorithm_params)
            elif algorithm.lower() == "regex":
                result_df, templates = self._regex_parse(result_df, content_column, algorithm_params)
            else:
                logger.warning(f"不支持的解析算法: {algorithm}，使用Drain算法")
                result_df, templates = self._drain_parse(result_df, content_column, algorithm_params)

            # 创建结果摘要
            summary = {
                "total_logs": len(logs_df),
                "parsed_templates": len(templates),
                "templates": templates
            }

            logger.info(f"日志解析完成: 识别出 {len(templates)} 个日志模板")
            return result_df, summary

        except Exception as e:
            logger.error(f"解析日志时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return logs_df, {"status": "error", "message": str(e)}

    def _drain_parse(
            self,
            df: pd.DataFrame,
            content_column: str,
            params: Dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        使用Drain算法解析日志模板

        Args:
            df: 日志DataFrame
            content_column: 内容列名
            params: 算法参数

        Returns:
            (处理后的DataFrame, 模板列表)
        """
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig

        # 配置Drain3
        config = TemplateMinerConfig()
        config.drain_sim_th = params.get("sim_th", 0.4)
        config.drain_depth = params.get("depth", 4)
        config.drain_max_children = params.get("max_children", 100)

        # 创建模板挖掘器
        template_miner = TemplateMiner(config=config)

        # 提取模板
        templates = []
        template_ids = []

        for _, log in df.iterrows():
            result = template_miner.add_log_message(log[content_column])
            template_ids.append(result.cluster_id)

            # 如果是新模板，添加到列表
            if result.is_new_cluster:
                templates.append(result.template_mined)

        # 更新DataFrame
        df["template"] = df[content_column].apply(lambda x: template_miner.match(x).get_template())
        df["template_id"] = template_ids

        return df, templates

    def _regex_parse(
            self,
            df: pd.DataFrame,
            content_column: str,
            params: Dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        使用正则表达式解析日志模板

        Args:
            df: 日志DataFrame
            content_column: 内容列名
            params: 算法参数

        Returns:
            (处理后的DataFrame, 模板列表)
        """
        # 常见的变量模式
        patterns = [
            (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>'),  # IP地址
            (r'\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b', '<UUID>'),  # UUID
            (r'\b\d{4}-\d{2}-\d{2}\b', '<DATE>'),  # 日期
            (r'\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b', '<TIME>'),  # 时间
            (r'(?<=[^A-Za-z0-9])(\d+)(?=[^A-Za-z0-9]|$)', '<NUM>'),  # 数字
            (r'\b([a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*)', '<PATH>'),  # Windows路径
            (r'\b((?:/[^/:*?"<>|\r\n]+)+/?)', '<PATH>'),  # Unix路径
            (r'([\'"])(?:(?=(\\?))\2.)*?\1', '<STR>')  # 字符串
        ]

        # 模板和ID映射
        template_to_id = {}
        next_id = 1

        # 创建结果列
        templates = []
        template_ids = []
        df_templates = []

        for _, row in df.iterrows():
            content = row[content_column]

            # 应用所有模式替换变量
            template = content
            for pattern, replacement in patterns:
                template = re.sub(pattern, replacement, template)

            # 存储模板
            if template not in template_to_id:
                template_to_id[template] = next_id
                next_id += 1
                templates.append(template)

            template_id = template_to_id[template]
            template_ids.append(template_id)
            df_templates.append(template)

        # 更新DataFrame
        df["template"] = df_templates
        df["template_id"] = template_ids

        return df, templates

    def detect_anomalies(
            self,
            df: pd.DataFrame,
            template_column: str = "template",
            algorithm: str = "isolation_forest",
            algorithm_params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        检测日志异常

        Args:
            df: 解析后的DataFrame
            template_column: 模板列名
            algorithm: 异常检测算法
            algorithm_params: 算法参数

        Returns:
            标记了异常的DataFrame和检测结果
        """
        if df.empty or template_column not in df.columns:
            logger.warning(f"没有有效的模板数据用于异常检测, 缺少列: {template_column}")
            return df, {"status": "no_data"}

        # 设置默认算法参数
        if algorithm_params is None:
            algorithm_params = {
                "contamination": 0.05,  # 预期异常比例
                "random_state": 42
            }

        try:
            # 创建特征
            vectorizer = CountVectorizer()
            # 使用模板作为特征
            X = vectorizer.fit_transform(df[template_column])

            # 使用隔离森林进行异常检测
            if algorithm.lower() == "isolation_forest":
                model = IsolationForest(
                    contamination=algorithm_params.get("contamination", 0.05),
                    random_state=algorithm_params.get("random_state", 42),
                    n_estimators=algorithm_params.get("n_estimators", 100)
                )

                # 预测异常
                predictions = model.fit_predict(X)

                # 转换预测结果 (-1 表示异常, 1 表示正常)
                # 将-1转换为1（异常），1转换为0（正常）
                anomalies = np.where(predictions == -1, 1, 0)

            else:
                logger.warning(f"不支持的异常检测算法: {algorithm}，使用隔离森林")
                model = IsolationForest(contamination=0.05, random_state=42)
                predictions = model.fit_predict(X)
                anomalies = np.where(predictions == -1, 1, 0)

            # 更新DataFrame
            result_df = df.copy()
            result_df["prediction"] = anomalies

            # 计算异常统计信息
            anomaly_count = np.sum(anomalies)
            anomaly_ratio = float(anomaly_count) / len(df) if len(df) > 0 else 0

            # 创建结果摘要
            summary = {
                "total_logs": len(df),
                "anomaly_count": int(anomaly_count),
                "anomaly_ratio": float(anomaly_ratio),
                "algorithm": algorithm,
                "parameters": algorithm_params
            }

            logger.info(f"异常检测完成: 检测到 {summary['anomaly_count']} 条异常日志，"
                        f"异常比例 {summary['anomaly_ratio']:.4f}")
            return result_df, summary

        except Exception as e:
            logger.error(f"检测异常时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return df, {"status": "error", "message": str(e)}

    def cluster_logs(
            self,
            df: pd.DataFrame,
            feature_column: str = "template",
            algorithm: str = "kmeans",
            algorithm_params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        聚类日志

        Args:
            df: 解析后的DataFrame
            feature_column: 特征列名
            algorithm: 聚类算法
            algorithm_params: 算法参数

        Returns:
            添加了聚类标签的DataFrame和聚类结果
        """
        if df.empty or feature_column not in df.columns:
            logger.warning(f"没有有效的特征数据用于聚类, 缺少列: {feature_column}")
            return df, {"status": "no_data"}

        # 设置默认算法参数
        if algorithm_params is None:
            # 自动确定聚类数量，但最多不超过20个聚类
            n_clusters = min(20, len(df) // 100 + 5)
            algorithm_params = {
                "n_clusters": n_clusters,
                "random_state": 42
            }

        try:
            # 创建特征
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(df[feature_column])

            # 使用KMeans聚类
            if algorithm.lower() == "kmeans":
                model = KMeans(
                    n_clusters=algorithm_params.get("n_clusters", 10),
                    random_state=algorithm_params.get("random_state", 42),
                    n_init=algorithm_params.get("n_init", 10)
                )

                # 执行聚类
                clusters = model.fit_predict(X)

            else:
                logger.warning(f"不支持的聚类算法: {algorithm}，使用KMeans")
                model = KMeans(n_clusters=10, random_state=42)
                clusters = model.fit_predict(X)

            # 更新DataFrame
            result_df = df.copy()
            result_df["cluster_id"] = clusters

            # 分析聚类结果
            cluster_counts = Counter(clusters)

            # 创建结果摘要
            summary = {
                "total_logs": len(df),
                "cluster_count": len(cluster_counts),
                "algorithm": algorithm,
                "parameters": algorithm_params,
                "cluster_distribution": {str(k): v for k, v in cluster_counts.items()}
            }

            logger.info(f"聚类完成: 形成 {summary['cluster_count']} 个聚类")
            return result_df, summary

        except Exception as e:
            logger.error(f"聚类日志时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return df, {"status": "error", "message": str(e)}

    def extract_and_process(
            self,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            max_logs: int = 100000,
            query_filter: Optional[Dict] = None,
            save_to_file: bool = True,
            save_to_es: bool = True,
            target_index: Optional[str] = None,
            preprocess: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        提取并处理日志的便捷方法

        Args:
            start_time: 开始时间
            end_time: 结束时间
            max_logs: 最大提取日志数
            query_filter: 额外的查询过滤条件
            save_to_file: 是否保存结果到文件
            save_to_es: 是否保存结果到ES
            target_index: 目标ES索引名称
            preprocess: 是否进行预处理

        Returns:
            处理后的DataFrame和处理结果信息
        """
        # 如果没有指定开始时间，使用上次处理的时间戳
        if not start_time and self.last_processed_timestamp:
            try:
                start_time = datetime.datetime.fromisoformat(self.last_processed_timestamp)
                logger.info(f"使用上次处理的时间戳作为开始时间: {start_time}")
            except ValueError:
                logger.warning(f"无法解析上次处理的时间戳: {self.last_processed_timestamp}")

        # 提取日志
        logs_df = self.es_connector.extract_logs(
            start_time=start_time,
            end_time=end_time,
            max_logs=max_logs,
            query_filter=query_filter
        )

        if logs_df.empty:
            logger.warning("没有日志数据需要处理")
            return logs_df, {"status": "no_data"}

        start_process_time = datetime.datetime.now()

        # 预处理
        if preprocess:
            # 导入预处理器
            from processing.preprocessor import LogPreprocessor
            preprocessor = LogPreprocessor()

            logs_df = preprocessor.preprocess(logs_df)

            # 使用更通用的异常处理
            logs_df = preprocessor.normalize_exceptions(logs_df)
            logger.info("执行了异常日志标准化")

        # 解析日志
        parsed_df, parsing_summary = self.parse_logs(logs_df)

        # 检测异常
        anomaly_df, anomaly_summary = self.detect_anomalies(parsed_df)

        # 聚类分析
        clustered_df, clustering_summary = self.cluster_logs(anomaly_df)

        # 提取错误模式
        error_patterns = self.extract_error_patterns(clustered_df)

        end_process_time = datetime.datetime.now()
        process_duration = (end_process_time - start_process_time).total_seconds()

        # 更新最后处理的时间戳
        if not clustered_df.empty and "@timestamp" in clustered_df.columns:
            self.last_processed_timestamp = clustered_df["@timestamp"].max()
            self.save_last_timestamp()

        # 创建完整的处理结果摘要
        processing_summary = {
            "extraction": {
                "total_logs": len(logs_df),
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None
            },
            "processing": {
                "start_time": start_process_time.isoformat(),
                "end_time": end_process_time.isoformat(),
                "duration_seconds": process_duration
            },
            "parsing": parsing_summary,
            "anomaly_detection": anomaly_summary,
            "clustering": clustering_summary,
            "error_patterns": error_patterns,
            "processing_time": datetime.datetime.now().isoformat()
        }

        # 保存结果到文件
        output_file = None
        if save_to_file:
            output_file = self.save_results(clustered_df, processing_summary)
            processing_summary["output_file"] = output_file

        # 保存结果到ES
        if save_to_es and target_index:
            # 如果未指定目标索引，使用默认值
            if target_index is None:
                target_index = "logai-processed"

            total, success = self.target_es_connector.save_to_new_index(
                clustered_df,
                target_index=target_index
            )
            processing_summary["es_export"] = {
                "total": total,
                "success": success,
                "target_index": target_index
            }

        return clustered_df, processing_summary

    def extract_error_patterns(
            self,
            df: pd.DataFrame,
            log_level_column: str = "log_level",
            error_levels: List[str] = ["ERROR", "FATAL"],
            message_column: str = "message",
            stack_trace_column: Optional[str] = "stack_trace"
    ) -> Dict[str, Any]:
        """
        提取错误模式

        Args:
            df: 日志DataFrame
            log_level_column: 日志级别列名
            error_levels: 错误级别列表
            message_column: 消息列名
            stack_trace_column: 堆栈跟踪列名

        Returns:
            错误模式摘要
        """
        if df.empty:
            return {"status": "no_data"}

        # 过滤错误日志
        if log_level_column in df.columns:
            error_df = df[df[log_level_column].isin(error_levels)]
        else:
            # 尝试从消息中推断错误
            error_keywords = ["error", "exception", "fail", "timeout", "unable to"]
            error_pattern = "|".join(error_keywords)
            error_df = df[df[message_column].str.lower().str.contains(error_pattern, na=False)]

        if error_df.empty:
            return {"status": "no_errors", "count": 0}

        # 按模板和聚类分组
        groupby_columns = []
        if "template" in error_df.columns:
            groupby_columns.append("template")
        if "template_id" in error_df.columns:
            groupby_columns.append("template_id")
        if "cluster_id" in error_df.columns:
            groupby_columns.append("cluster_id")

        if not groupby_columns:
            # 如果没有模板或聚类信息，使用消息的前50个字符作为分组依据
            error_df["error_prefix"] = error_df[message_column].str[:50]
            groupby_columns = ["error_prefix"]

        # 分组统计
        try:
            error_patterns = error_df.groupby(groupby_columns).size().reset_index(name="count")
            error_patterns = error_patterns.sort_values("count", ascending=False)
        except Exception as e:
            logger.error(f"分组统计错误模式时出错: {str(e)}")
            return {"status": "error", "message": str(e), "count": len(error_df)}

        # 为每种错误模式提取示例
        patterns_with_examples = []

        for _, pattern in error_patterns.iterrows():
            # 过滤匹配此模式的记录
            filter_conditions = {}
            for col in groupby_columns:
                filter_conditions[col] = pattern[col]

            # 应用过滤条件
            match_df = error_df.copy()
            for col, val in filter_conditions.items():
                match_df = match_df[match_df[col] == val]

            # 获取一个示例
            if not match_df.empty:
                example = match_df.iloc[0]
                example_dict = {
                    "pattern": {col: str(pattern[col]) for col in groupby_columns},
                    "count": int(pattern["count"]),
                    "example_message": str(example[message_column])
                }

                # 如果有堆栈跟踪，添加堆栈信息
                if stack_trace_column and stack_trace_column in example and example[stack_trace_column]:
                    stack_trace = example[stack_trace_column]
                    if stack_trace and not pd.isna(stack_trace):
                        # 限制堆栈跟踪长度
                        max_stack_length = 1000  # 字符数
                        if len(str(stack_trace)) > max_stack_length:
                            stack_trace = str(stack_trace)[:max_stack_length] + "..."
                        example_dict["example_stack_trace"] = str(stack_trace)

                patterns_with_examples.append(example_dict)

        return {
            "status": "success",
            "count": len(error_df),
            "unique_patterns": len(patterns_with_examples),
            "patterns": patterns_with_examples
        }

    def process_incremental(
            self,
            time_window: datetime.timedelta = datetime.timedelta(hours=1),
            max_logs: int = 10000,
            save_to_file: bool = True,
            save_to_es: bool = True,
            source_index_pattern: Optional[str] = None,
            target_index: Optional[str] = None
    ):
        """增量处理日志"""
        # 获取配置
        config = get_config()

        # 确定源索引模式和目标索引
        source_pattern = source_index_pattern or config.get(
            "elasticsearch.index_pattern") or self.es_connector.es_index_pattern
        target_idx = target_index or config.get("target_elasticsearch.index", "logai-processed")

        # 确定结束时间
        end_time = datetime.datetime.now()

        # 查询源ES的最新时间戳
        source_latest = self._get_latest_timestamp(source_pattern, is_source=True)  # 使用源ES连接器

        # 查询目标ES的最新时间戳
        target_latest = self._get_latest_timestamp(target_idx, is_source=False)  # 使用目标ES连接器

        # 确定开始时间
        if target_latest:
            # 从目标ES最新记录的时间开始
            start_time = target_latest
            logger.info(f"从目标索引最新时间戳开始: {start_time}")
        else:
            # 如果目标ES没有记录，使用时间窗口
            start_time = end_time - time_window
            logger.info(f"目标索引无记录，使用时间窗口: {start_time}")

        # 如果源ES没有比目标ES更新的数据，无需处理
        if source_latest and target_latest and source_latest <= target_latest:
            logger.info("源索引无新数据，跳过处理")
            return pd.DataFrame(), {"status": "no_new_data"}

        # 执行处理
        return self.extract_and_process(
            start_time=start_time,
            end_time=end_time,
            max_logs=max_logs,
            save_to_file=save_to_file,
            save_to_es=save_to_es,
            target_index=target_idx
        )

    def _get_latest_timestamp(self, index_pattern: str, is_source: bool = False) -> Optional[datetime.datetime]:
        try:
            connector = self.es_connector if is_source else self.target_es_connector
            config = get_config()

            # 获取配置的时间戳字段名
            timestamp_field = config.get("field_mappings.timestamp_field", "@timestamp")

            # 如果索引不存在则返回None
            if not connector.es_client.indices.exists(index=index_pattern):
                logger.info(f"索引 {index_pattern} 不存在")
                return None

            query = {
                "size": 1,
                "sort": [{timestamp_field: {"order": "desc"}}],
                "_source": [timestamp_field]
            }

            result = connector.es_client.search(
                index=index_pattern,
                body=query
            )

            if result["hits"]["hits"]:
                timestamp_str = result["hits"]["hits"][0]["_source"][timestamp_field]
                return datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            return None
        except Exception as e:
            logger.warning(f"获取索引 {index_pattern} 最新时间戳失败: {str(e)}")
            return None

    def save_results(
            self,
            processed_df: pd.DataFrame,
            summary: Dict,
            output_file: Optional[str] = None
    ) -> str:
        """
        保存处理结果

        Args:
            processed_df: 处理后的DataFrame
            summary: 处理结果摘要
            output_file: 输出文件路径，如果为None则自动生成

        Returns:
            保存的文件路径
        """
        if processed_df.empty:
            logger.warning("没有结果需要保存")
            return ""

        # 如果没有指定输出文件，则生成一个基于时间的文件名
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"log_results_{timestamp}.json")

        try:
            # 将DataFrame转换为可序列化的格式
            # 处理特殊类型
            def serialize_value(v):
                if isinstance(v, (datetime.datetime, pd.Timestamp)):
                    return v.isoformat()
                elif isinstance(v, (pd.Series, pd.DataFrame)):
                    return str(v)
                elif hasattr(v, 'tolist'):  # numpy arrays
                    return v.tolist()
                else:
                    return v

            # 转换DataFrame为可序列化的字典
            records = []
            for _, row in processed_df.iterrows():
                record = {}
                for col, val in row.items():
                    record[col] = serialize_value(val)
                records.append(record)

            result_dict = {
                "metadata": summary,
                "data": records
            }

            # 保存到文件
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"处理结果已保存至: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

        def process_specific_errors(
                self,
                df: pd.DataFrame,
                error_type: Optional[str] = None,
                module_name: Optional[str] = None,
                error_message_contains: Optional[str] = None
        ) -> pd.DataFrame:
            """
            处理特定类型的错误

            Args:
                df: 日志DataFrame
                error_type: 错误类型，如'System.NullReferenceException'
                module_name: 模块名称
                error_message_contains: 错误消息包含的文本

            Returns:
                过滤后的DataFrame
            """
            if df.empty:
                return df

            filtered_df = df.copy()

            # 按错误类型过滤
            if error_type and "exception_type" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["exception_type"].str.contains(error_type, na=False)]

            # 按模块名称过滤
            if module_name and "exception_class" in filtered_df.columns:
                # 从异常类名中提取模块
                filtered_df["module"] = filtered_df["exception_class"].str.split(".").str[:-1].str.join(".")
                filtered_df = filtered_df[filtered_df["module"].str.contains(module_name, na=False)]

            # 按错误消息过滤
            if error_message_contains and "message" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["message"].str.contains(error_message_contains, na=False)]

            return filtered_df

        def get_cluster_summary(
                self,
                df: pd.DataFrame,
                cluster_id_column: str = "cluster_id"
        ) -> Dict[str, Any]:
            """
            获取聚类摘要信息

            Args:
                df: 处理后的DataFrame
                cluster_id_column: 聚类ID列名

            Returns:
                聚类摘要
            """
            if df.empty or cluster_id_column not in df.columns:
                return {"status": "no_data"}

            clusters = {}

            # 按聚类ID分组
            for cluster_id, group in df.groupby(cluster_id_column):
                # 获取此聚类的模板
                template = None
                if "template" in group.columns:
                    # 使用出现次数最多的模板
                    template = group["template"].value_counts().index[0]

                # 统计日志级别分布
                level_distribution = {}
                if "log_level" in group.columns:
                    level_distribution = group["log_level"].value_counts().to_dict()

                # 获取示例消息
                example_message = None
                if "message" in group.columns:
                    example_message = group["message"].iloc[0]

                # 时间范围
                time_range = {}
                if "@timestamp" in group.columns:
                    time_range = {
                        "earliest": group["@timestamp"].min(),
                        "latest": group["@timestamp"].max()
                    }

                # 是否包含异常
                contains_errors = False
                if "log_level" in group.columns:
                    contains_errors = any(level in ["ERROR", "FATAL"] for level in group["log_level"])

                clusters[str(cluster_id)] = {
                    "count": len(group),
                    "template": template,
                    "level_distribution": level_distribution,
                    "example_message": example_message,
                    "time_range": time_range,
                    "contains_errors": contains_errors
                }

            return {
                "status": "success",
                "total_clusters": len(clusters),
                "clusters": clusters
            }

        def extract_template_variables(
                self,
                df: pd.DataFrame,
                template_column: str = "template",
                message_column: str = "message"
        ) -> pd.DataFrame:
            """
            从日志中提取模板变量

            Args:
                df: 日志DataFrame
                template_column: 模板列名
                message_column: 消息列名

            Returns:
                添加了变量列的DataFrame
            """
            if df.empty or template_column not in df.columns or message_column not in df.columns:
                return df

            # 创建结果DataFrame
            result_df = df.copy()
            result_df["template_variables"] = None

            # 对每个模板，创建正则表达式来提取变量
            template_patterns = {}

            for _, row in df.iterrows():
                template = row[template_column]
                message = row[message_column]

                # 如果模板尚未处理
                if template not in template_patterns:
                    # 转义正则表达式特殊字符
                    pattern_str = re.escape(template)

                    # 替换变量占位符为捕获组
                    pattern_str = pattern_str.replace("\\<NUM\\>", "([0-9]+)")
                    pattern_str = pattern_str.replace("\\<IP\\>", "([0-9.]+)")
                    pattern_str = pattern_str.replace("\\<PATH\\>", "([^\\s]+)")
                    pattern_str = pattern_str.replace("\\<DATE\\>", "([0-9-]+)")
                    pattern_str = pattern_str.replace("\\<TIME\\>", "([0-9:]+(?:\\.[0-9]+)?)")
                    pattern_str = pattern_str.replace("\\<UUID\\>", "([0-9a-f-]+)")
                    pattern_str = pattern_str.replace("\\<STR\\>", "([^\\s]+)")

                    # 编译正则表达式
                    template_patterns[template] = re.compile(f"^{pattern_str}$")

                # 提取变量
                pattern = template_patterns[template]
                match = pattern.match(message)

                if match:
                    variables = match.groups()
                    result_df.loc[_, "template_variables"] = json.dumps(variables)

            return result_df