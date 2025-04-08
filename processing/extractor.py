# processing/extractor.py
import logging
import json
import os
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd

from logai.applications.log_parsing.parsing_handler import LogParsingHandler
from logai.applications.anomaly_detection.anomaly_detection_handler import AnomalyDetectionHandler
from logai.analysis.clustering import LogClustering

from data.es_connector import ESConnector
from processing.preprocessor import LogPreprocessor
from config import get_config, get_model_config

logger = logging.getLogger(__name__)


class LogProcessor:
    """日志处理器，整合了LogAI功能"""

    def __init__(
            self,
            es_connector: Optional[ESConnector] = None,
            preprocessor: Optional[LogPreprocessor] = None,
            output_dir: Optional[str] = None,
            create_output_dir: bool = True
    ):
        """
        初始化日志处理器

        Args:
            es_connector: ES连接器，如果为None则自动创建
            preprocessor: 日志预处理器，如果为None则自动创建
            output_dir: 输出目录，如果为None则使用配置中的值
            create_output_dir: 是否创建输出目录
        """
        # 加载配置
        self.config = get_config()
        self.model_config = get_model_config()

        # 设置输出目录
        self.output_dir = output_dir or self.config.get("app.output_dir", "./output")

        # 创建输出目录
        if create_output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建输出目录: {self.output_dir}")

        # 初始化组件
        self.es_connector = es_connector or self._create_es_connector()
        self.preprocessor = preprocessor or LogPreprocessor()

        # 状态跟踪
        self.last_processed_timestamp = None

    def _create_es_connector(self) -> ESConnector:
        """
        创建ES连接器

        Returns:
            ES连接器实例
        """
        es_config = self.config.get("elasticsearch", {})

        connector = ESConnector(
            es_host=es_config.get("host", "localhost"),
            es_port=es_config.get("port", 9200),
            es_user=es_config.get("username"),
            es_password=es_config.get("password"),
            es_index_pattern=es_config.get("index_pattern", "logstash-*"),
            use_ssl=es_config.get("use_ssl", False),
            verify_certs=es_config.get("verify_certs", False)
        )

        return connector

    def parse_logs(
            self,
            logs_df: pd.DataFrame,
            content_column: str = "message",
            timestamp_column: str = "@timestamp",
            id_column: str = "_id",
            algorithm: Optional[str] = None,
            algorithm_params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        解析日志内容，提取模板

        Args:
            logs_df: 包含日志数据的DataFrame
            content_column: 内容列名
            timestamp_column: 时间戳列名
            id_column: ID列名
            algorithm: 解析算法，如果为None则使用配置
            algorithm_params: 算法参数，如果为None则使用配置

        Returns:
            解析后的DataFrame和解析结果
        """
        if logs_df.empty:
            logger.warning("没有日志数据需要解析")
            return logs_df, {"status": "no_data"}

        logger.info(f"开始解析 {len(logs_df)} 条日志记录")

        # 确保列名符合LogAI要求
        df_copy = logs_df.copy()

        # 重命名列
        column_mapping = {
            content_column: "content",
            timestamp_column: "timestamp",
            id_column: "log_id"
        }

        # 只重命名存在的列
        for orig_col, new_col in column_mapping.items():
            if orig_col in df_copy.columns:
                df_copy = df_copy.rename(columns={orig_col: new_col})

        # 确保必需的列存在
        if "content" not in df_copy.columns:
            logger.error(f"找不到内容列 '{content_column}'")
            return logs_df, {"status": "error", "message": f"找不到内容列 '{content_column}'"}

        # 获取算法和参数
        if algorithm is None or algorithm_params is None:
            parse_config = self.model_config.get_parse_algorithm()
            algorithm = algorithm or parse_config["algorithm"]
            algorithm_params = algorithm_params or parse_config["params"]

        try:
            # 创建解析处理器
            parsing_handler = LogParsingHandler(
                parsing_algorithm=algorithm,
                parsing_algorithm_params=algorithm_params
            )

            # 执行解析
            parsed_result = parsing_handler.analyze(df_copy)

            # 获取解析后的数据
            result_df = parsed_result.processed_data

            # 获取模板信息
            templates = parsed_result.get_templates()

            # 创建结果摘要
            summary = {
                "total_logs": len(logs_df),
                "parsed_templates": len(templates),
                "templates": templates
            }

            # 将结果合并回原始DataFrame
            # 重命名解析后的列以避免冲突
            result_columns = {
                "parsed_content": "template",
                "cluster_id": "template_id"
            }

            result_df = result_df.rename(columns=result_columns)

            # 恢复原始列名
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            for new_col, orig_col in reverse_mapping.items():
                if new_col in result_df.columns:
                    result_df = result_df.rename(columns={new_col: orig_col})

            logger.info(f"日志解析完成: 识别出 {summary['parsed_templates']} 个日志模板")
            return result_df, summary

        except Exception as e:
            logger.error(f"解析日志时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return logs_df, {"status": "error", "message": str(e)}

    def detect_anomalies(
            self,
            df: pd.DataFrame,
            template_column: str = "template",
            algorithm: Optional[str] = None,
            algorithm_params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        检测日志异常

        Args:
            df: 解析后的DataFrame
            template_column: 模板列名
            algorithm: 异常检测算法，如果为None则使用配置
            algorithm_params: 算法参数，如果为None则使用配置

        Returns:
            标记了异常的DataFrame和检测结果
        """
        if df.empty or template_column not in df.columns:
            logger.warning(f"没有有效的模板数据用于异常检测, 缺少列: {template_column}")
            return df, {"status": "no_data"}

        # 获取算法和参数
        if algorithm is None or algorithm_params is None:
            anomaly_config = self.model_config.get_anomaly_detection_algorithm()
            algorithm = algorithm or anomaly_config["algorithm"]
            algorithm_params = algorithm_params or anomaly_config["params"]

        try:
            # 创建异常检测处理器
            df_copy = df.copy()

            # 重命名列以符合LogAI要求
            df_copy = df_copy.rename(columns={template_column: "feature"})

            anomaly_handler = AnomalyDetectionHandler(
                detection_algorithm=algorithm,
                detection_algorithm_params=algorithm_params
            )

            # 执行异常检测
            anomaly_result = anomaly_handler.analyze(df_copy)

            # 获取检测结果
            result_df = anomaly_result.processed_data

            # 恢复原始列名
            result_df = result_df.rename(columns={"feature": template_column})

            # 计算异常统计信息
            anomaly_count = result_df["prediction"].sum()
            anomaly_ratio = anomaly_count / len(result_df) if len(result_df) > 0 else 0

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
            algorithm: Optional[str] = None,
            algorithm_params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        聚类日志

        Args:
            df: 解析后的DataFrame
            feature_column: 特征列名
            algorithm: 聚类算法，如果为None则使用配置
            algorithm_params: 算法参数，如果为None则使用配置

        Returns:
            添加了聚类标签的DataFrame和聚类结果
        """
        if df.empty or feature_column not in df.columns:
            logger.warning(f"没有有效的特征数据用于聚类, 缺少列: {feature_column}")
            return df, {"status": "no_data"}

        # 获取算法和参数
        if algorithm is None or algorithm_params is None:
            clustering_config = self.model_config.get_clustering_algorithm()
            algorithm = algorithm or clustering_config["algorithm"]
            algorithm_params = algorithm_params or clustering_config["params"]

            # 自动调整聚类数量
            if "n_clusters" in algorithm_params and algorithm_params["n_clusters"] > len(df) // 2:
                # 避免聚类数量过多
                algorithm_params["n_clusters"] = max(5, min(20, len(df) // 10))
                logger.info(f"自动调整聚类数量为: {algorithm_params['n_clusters']}")

        try:
            # 创建聚类分析器
            clustering = LogClustering(
                algorithm=algorithm,
                algorithm_params=algorithm_params
            )

            # 执行聚类
            clustering_result = clustering.fit_predict(
                df,
                feature_col=feature_column
            )

            # 合并结果
            if "cluster" in clustering_result:
                result_df = df.copy()
                result_df["cluster_id"] = clustering_result["cluster"]

                # 分析聚类结果
                cluster_counts = result_df["cluster_id"].value_counts().to_dict()

                # 创建结果摘要
                summary = {
                    "total_logs": len(df),
                    "cluster_count": len(cluster_counts),
                    "algorithm": algorithm,
                    "parameters": algorithm_params,
                    "cluster_distribution": cluster_counts
                }

                logger.info(f"聚类完成: 形成 {summary['cluster_count']} 个聚类")
                return result_df, summary
            else:
                logger.warning("聚类结果中没有'cluster'列")
                return df, {"status": "error", "message": "聚类结果中没有'cluster'列"}

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
            save_to_es: bool = False,
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
            start_time = datetime.datetime.fromisoformat(self.last_processed_timestamp)
            logger.info(f"使用上次处理的时间戳作为开始时间: {start_time}")

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
            logs_df = self.preprocessor.preprocess(logs_df)

            # 处理ABP特定日志（如果有ABP日志）
            if any(logs_df["message"].str.contains("Volo.Abp", na=False)):
                logs_df = self.preprocessor.normalize_abp_logs(logs_df)
                logger.info("执行了ABP日志标准化")

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
            # 如果未指定目标索引，使用配置中的值
            if target_index is None:
                target_index = self.config.get("target_elasticsearch.index", "logai-processed")

            total, success = self.es_connector.save_to_new_index(
                clustered_df,
                target_index=target_index
            )
            processing_summary["es_export"] = {
                "total": total,
                "success": success,
                "target_index": target_index
            }

        return clustered_df, processing_summary

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
            output_file = os.path.join(self.output_dir, f"logai_results_{timestamp}.json")

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
            error_type: 错误类型，如'Volo.Abp.AbpInitializationException'
            module_name: 模块名称，如'Uow'
            error_message_contains: 错误消息包含的文本

        Returns:
            过滤后的DataFrame
        """
        if df.empty:
            return df

        filtered_df = df.copy()

        # 按错误类型过滤
        if error_type and "abp_exception_type" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["abp_exception_type"].str.contains(error_type, na=False)]

        # 按模块名称过滤
        if module_name and "abp_module" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["abp_module"] == module_name]

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

    def process_incremental(
            self,
            time_window: datetime.timedelta = datetime.timedelta(hours=1),
            max_logs: int = 10000,
            save_to_file: bool = True,
            save_to_es: bool = True,
            target_index: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        增量处理日志

        Args:
            time_window: 时间窗口
            max_logs: 每次处理的最大日志数
            save_to_file: 是否保存结果到文件
            save_to_es: 是否保存结果到ES
            target_index: 目标ES索引名称

        Returns:
            处理后的DataFrame和处理结果信息
        """
        # 确定开始时间和结束时间
        end_time = datetime.datetime.now()

        if self.last_processed_timestamp:
            # 使用上次处理的时间作为开始时间
            start_time = datetime.datetime.fromisoformat(self.last_processed_timestamp)
        else:
            # 使用时间窗口计算开始时间
            start_time = end_time - time_window

        logger.info(f"增量处理日志 - 时间范围: {start_time} 至 {end_time}")

        # 执行处理
        return self.extract_and_process(
            start_time=start_time,
            end_time=end_time,
            max_logs=max_logs,
            save_to_file=save_to_file,
            save_to_es=save_to_es,
            target_index=target_index
        )


# processing/__init__.py
from .preprocessor import LogPreprocessor
from .extractor import LogProcessor

__all__ = ['LogPreprocessor', 'LogProcessor']