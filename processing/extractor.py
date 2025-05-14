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
from deduplication import LogDeduplicator


from config import get_config, get_model_config

logger = get_logger(__name__)


class LogProcessor:
    """日志处理器，负责日志解析、异常检测和聚类"""

    def __init__(
            self,
            es_connector: ESConnector,
            target_es_connector: ESConnector,
            output_dir: str = "./output",
            create_output_dir: bool = True,
            field_mappings: Optional[Dict[str, str]] = None
    ):
        """
        初始化日志处理器

        Args:
            es_connector: ES连接器
            target_es_connector: 目标ES连接器
            output_dir: 输出目录
            create_output_dir: 是否创建输出目录
            field_mappings: 字段映射配置
        """
        self.es_connector = es_connector
        self.target_es_connector = target_es_connector
        self.output_dir = output_dir

        # 设置字段映射
        self.field_mappings = field_mappings or {}
        self.message_field = self.field_mappings.get("message_field", "message")
        self.level_field = self.field_mappings.get("level_field", "log_level")
        self.service_field = self.field_mappings.get("service_field", "service_name")
        # 注意：timestamp_field 仍然使用 @timestamp

        # 创建输出目录 - 使用 exist_ok=True 确保不会因为目录已存在而报错
        if create_output_dir:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"创建或确认输出目录存在: {output_dir}")
            except Exception as e:
                logger.error(f"创建输出目录失败: {str(e)}")

        # 状态跟踪
        self.last_processed_timestamp = None

        # 加载和保存最近处理的时间戳
        self.timestamp_file = os.path.join(self.output_dir, "last_timestamp.txt")
        self.load_last_timestamp()

        # 初始化模板存储
        self.template_store = {}
        self.cluster_store = {}

        # 初始化日志去重器
        self.deduplicator = LogDeduplicator(
            similarity_threshold=0.92,
            time_window_hours=2,
            min_cluster_size=2,
            novelty_threshold=0.05,
            keep_latest_count=3,
            keep_anomalies=True
        )

        logger.info("日志处理器初始化完成")

    def save_last_timestamp(self):
        """保存最后处理的时间戳到文件"""
        if self.last_processed_timestamp:
            try:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(self.timestamp_file), exist_ok=True)

                # 转换为字符串（如果是Timestamp对象）
                timestamp_str = self.last_processed_timestamp
                if hasattr(self.last_processed_timestamp, 'isoformat'):
                    timestamp_str = self.last_processed_timestamp.isoformat()

                with open(self.timestamp_file, "w") as f:
                    f.write(timestamp_str)
                logger.info(f"已保存最后处理时间戳: {timestamp_str}")
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
            content_column: str = None,
            timestamp_column: str = "@timestamp",
            id_column: str = "_id",
            algorithm: str = "drain",
            algorithm_params: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        解析日志内容，提取模板

        Args:
            logs_df: 包含日志数据的DataFrame
            content_column: 内容列名（如果为None则使用配置的message_field）
            timestamp_column: 时间戳列名
            id_column: ID列名
            algorithm: 解析算法，默认为'drain'
            algorithm_params: 算法参数

        Returns:
            解析后的DataFrame和解析结果
        """
        # 使用配置的字段名或默认值
        content_column = content_column or self.message_field

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

        # 调试信息
        first_result = None

        for idx, log in df.iterrows():
            try:
                result = template_miner.add_log_message(log[content_column])

                # 保存第一个结果用于调试
                if first_result is None:
                    first_result = result
                    logger.debug(f"Drain3 返回结果类型: {type(result)}")
                    if isinstance(result, dict):
                        logger.debug(f"Drain3 返回字典键: {list(result.keys())}")

                # 获取cluster_id，尝试不同的访问方式
                cluster_id = None
                if isinstance(result, dict):
                    # 尝试各种可能的键名
                    if "cluster_id" in result:
                        cluster_id = result["cluster_id"]
                    elif "cluster" in result:
                        cluster_id = result["cluster"]
                    elif "id" in result:
                        cluster_id = result["id"]
                    else:
                        # 如果找不到合适的键，使用索引作为ID
                        cluster_id = idx
                else:
                    # 尝试作为对象访问
                    try:
                        cluster_id = result.cluster_id
                    except AttributeError:
                        try:
                            cluster_id = result.cluster
                        except AttributeError:
                            try:
                                cluster_id = result.id
                            except AttributeError:
                                cluster_id = idx

                template_ids.append(cluster_id)

                # 获取是否是新模板
                is_new = False
                if isinstance(result, dict):
                    # 尝试各种可能的键名
                    if "is_new_cluster" in result:
                        is_new = result["is_new_cluster"]
                    elif "is_new" in result:
                        is_new = result["is_new"]
                    elif "new" in result:
                        is_new = result["new"]
                else:
                    # 尝试作为对象访问
                    try:
                        is_new = result.is_new_cluster
                    except AttributeError:
                        try:
                            is_new = result.is_new
                        except AttributeError:
                            try:
                                is_new = result.new
                            except AttributeError:
                                pass

                # 获取模板
                template = None
                if isinstance(result, dict):
                    # 尝试各种可能的键名
                    if "template_mined" in result:
                        template = result["template_mined"]
                    elif "template" in result:
                        template = result["template"]
                    elif "pattern" in result:
                        template = result["pattern"]
                else:
                    # 尝试作为对象访问
                    try:
                        template = result.template_mined
                    except AttributeError:
                        try:
                            template = result.template
                        except AttributeError:
                            try:
                                template = result.pattern
                            except AttributeError:
                                template = log[content_column]  # 使用原始消息作为后备

                # 如果是新模板，添加到列表
                if is_new and template:
                    templates.append(template)

            except Exception as e:
                logger.warning(f"处理日志 {idx} 时出错: {str(e)}")
                template_ids.append(idx)  # 使用索引作为后备ID

        # 直接使用每条日志内容作为模板（如果正常解析失败）
        if not templates:
            logger.warning("未能提取有效模板，使用日志内容作为模板")
            templates = df[content_column].unique().tolist()
            df["template"] = df[content_column]
            df["template_id"] = template_ids
        else:
            # 尝试安全地提取模板
            try:
                def extract_template(x):
                    try:
                        match_result = template_miner.match(x)
                        if isinstance(match_result, dict):
                            if "template" in match_result:
                                return match_result["template"]
                            elif "pattern" in match_result:
                                return match_result["pattern"]
                        else:
                            try:
                                return match_result.get_template()
                            except:
                                return match_result.template if hasattr(match_result, "template") else x
                    except:
                        return x  # 如果失败，返回原始文本

                df["template"] = df[content_column].apply(extract_template)
                df["template_id"] = template_ids
            except Exception as e:
                logger.error(f"提取模板时出错: {str(e)}")
                # 降级方案
                df["template"] = df[content_column]
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

    # 在 processing/extractor.py 中添加一个正则预筛选去重方法

    def prefilter_duplicates(
            self,
            df: pd.DataFrame,
            message_column: str = None,
            timestamp_column: str = "@timestamp",
            time_window_minutes: int = 60,
            pattern_column: str = "pattern_group"
    ) -> pd.DataFrame:
        """
        使用正则表达式和模式匹配对日志进行预筛选去重，
        这比向量化去重更高效，可以先快速过滤掉明显的重复日志。

        Args:
            df: 输入的DataFrame
            message_column: 消息列，如果为None则使用self.message_field
            timestamp_column: 时间戳列
            time_window_minutes: 时间窗口（分钟）
            pattern_column: 存储模式组的列名

        Returns:
            去重后的DataFrame
        """
        import re
        from collections import defaultdict

        # 如果DataFrame为空，直接返回
        if df.empty:
            return df

        # 使用配置的字段名或默认值
        message_column = message_column or self.message_field

        # 确保消息列存在
        if message_column not in df.columns:
            logger.warning(f"消息列 '{message_column}' 不在DataFrame中，跳过预筛选去重")
            return df

        # 记录原始数量
        original_count = len(df)
        logger.info(f"开始正则预筛选去重，原始记录数: {original_count}")

        # 1. 替换常见的变量模式
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

        # 创建一个新列存储模式化后的消息
        df[pattern_column] = df[message_column].copy()

        # 应用模式替换
        for pattern, replacement in patterns:
            df[pattern_column] = df[pattern_column].str.replace(pattern, replacement, regex=True)

        # 2. 按时间窗口和服务名称分组处理
        # 如果时间戳列存在，添加时间窗口分组标识
        if timestamp_column in df.columns:
            # 确保timestamp列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                try:
                    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                except:
                    logger.warning(f"无法将{timestamp_column}列转换为datetime类型，不进行时间窗口分组")

            # 添加时间窗口标识
            if pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                # 按时间窗口分组，例如每小时一组
                df['time_window'] = df[timestamp_column].dt.floor(f'{time_window_minutes}min')

        # 组合分组键，优先考虑服务名+时间窗口
        group_keys = []
        if self.service_field in df.columns:
            group_keys.append(self.service_field)
        if 'time_window' in df.columns:
            group_keys.append('time_window')

        # 如果没有可用的分组键，直接使用模式分组
        if not group_keys:
            logger.info("未找到服务名或时间窗口字段，仅使用模式去重")
            # 获取每个模式的第一次出现
            df_deduplicated = df.drop_duplicates(subset=[pattern_column])

            dedup_count = original_count - len(df_deduplicated)
            logger.info(f"正则预筛选去重完成: 从{original_count}条记录中去除了{dedup_count}条重复记录")

            # 清理临时列
            df_deduplicated = df_deduplicated.drop(columns=[pattern_column])
            return df_deduplicated

        # 按分组进行去重
        result_dfs = []

        for group_name, group_df in df.groupby(group_keys):
            # 获取每个分组内每个模式的第一次出现
            group_deduped = group_df.drop_duplicates(subset=[pattern_column])
            result_dfs.append(group_deduped)

        # 合并结果
        df_deduplicated = pd.concat(result_dfs)

        # 计算去重数量
        dedup_count = original_count - len(df_deduplicated)
        logger.info(f"正则预筛选去重完成: 从{original_count}条记录中去除了{dedup_count}条重复记录")

        # 清理临时列
        if 'time_window' in df_deduplicated.columns:
            df_deduplicated = df_deduplicated.drop(columns=['time_window'])

        df_deduplicated = df_deduplicated.drop(columns=[pattern_column])

        return df_deduplicated

    # 修改 extract_and_process 方法，添加正则预筛选去重步骤

    def extract_and_process(
            self,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            max_logs: int = 100000,
            query_filter: Optional[Dict] = None,
            save_to_file: bool = True,
            save_to_es: bool = True,
            target_index: Optional[str] = None,
            preprocess: bool = True,
            deduplication: bool = True,
            prefilter: bool = True,  # 新增参数：是否使用正则预筛选去重
            time_window_minutes: int = 30
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        提取并处理日志的便捷方法
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
        dedup_stats = {"original_count": len(logs_df)}

        # 预处理
        result_df = logs_df.copy()  # 创建一个初始结果DataFrame
        if preprocess:
            # 导入预处理器
            from processing.preprocessor import LogPreprocessor

            # 使用和LogProcessor相同的字段映射
            preprocessor = LogPreprocessor(field_mappings=self.field_mappings)

            # 使用配置的字段名
            result_df = preprocessor.preprocess(result_df, message_column=self.message_field)

            # 使用更通用的异常处理
            result_df = preprocessor.normalize_exceptions(result_df, message_column=self.message_field)
            logger.info("执行了异常日志标准化")

        # 正则预筛选去重 - 在向量化之前进行，节省资源
        if prefilter and deduplication:
            logger.info("开始使用正则预筛选去重...")
            prefilter_count = len(result_df)
            result_df = self.prefilter_duplicates(
                result_df,
                message_column=self.message_field,
                time_window_minutes=time_window_minutes  # 30分钟时间窗口
            )
            prefilter_reduced = prefilter_count - len(result_df)
            logger.info(f"正则预筛选去重减少了 {prefilter_reduced} 条记录")

            # 添加到统计信息
            dedup_stats["prefilter_count"] = prefilter_count
            dedup_stats["prefilter_reduced"] = prefilter_reduced
            dedup_stats["after_prefilter"] = len(result_df)

        # 如果启用去重，在解析日志前进行智能聚类去重
        if deduplication and not result_df.empty:
            original_count = len(result_df)
            dedup_stats["before_vector_dedup"] = original_count

            # 确保日志具有向量表示
            if 'vector' not in result_df.columns and hasattr(self, 'embedding_model'):
                # 调用嵌入模型获取向量
                from processing.vectorizer import LogVectorizer
                vectorizer = LogVectorizer(self.embedding_model)
                # 使用配置的消息字段而非硬编码的 'message'
                result_df = vectorizer.add_vectors_to_df(result_df, content_column=self.message_field)

            # 应用智能聚类去重
            if 'vector' in result_df.columns:
                result_df = self.deduplicator.smart_cluster_deduplication(result_df)
                dedup_count = original_count - len(result_df)
                logger.info(f"向量去重完成: 从{original_count}条日志中去除了{dedup_count}条重复日志")

                # 添加到统计信息
                dedup_stats["vector_dedup_reduced"] = dedup_count
                dedup_stats["after_vector_dedup"] = len(result_df)
            else:
                logger.warning("日志数据缺少向量表示，无法执行智能去重")

        # 解析日志 - 使用配置的消息字段
        parsed_df, parsing_summary = self.parse_logs(result_df, content_column=self.message_field)

        # 检测异常
        anomaly_df, anomaly_summary = self.detect_anomalies(parsed_df, template_column="template")

        # 聚类分析
        clustered_df, clustering_summary = self.cluster_logs(anomaly_df, feature_column="template")

        # 提取错误模式 - 使用配置的级别字段
        error_patterns = self.extract_error_patterns(
            clustered_df,
            log_level_column=self.level_field,
            message_column=self.message_field
        )

        end_process_time = datetime.datetime.now()
        process_duration = (end_process_time - start_process_time).total_seconds()

        # 更新最后处理的时间戳
        if not clustered_df.empty and "@timestamp" in clustered_df.columns:
            # 获取最大时间戳并确保它是字符串格式
            max_timestamp = clustered_df["@timestamp"].max()
            if hasattr(max_timestamp, 'isoformat'):
                self.last_processed_timestamp = max_timestamp.isoformat()
            else:
                self.last_processed_timestamp = str(max_timestamp)
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
            "deduplication": dedup_stats,
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
        if save_to_es:
            # 如果未指定目标索引，使用默认值
            if target_index is None:
                target_index = "logai-processed"

            # 确保索引存在并有正确的映射
            if not self.target_es_connector.es_client.indices.exists(index=target_index):
                # 创建索引并设置正确的IK分词器
                settings = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "ik_smart": {
                                    "type": "custom",
                                    "tokenizer": "ik_smart"
                                },
                                "ik_max_word": {
                                    "type": "custom",
                                    "tokenizer": "ik_max_word"
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "@timestamp": {"type": "date"},
                            "message": {
                                "type": "text",
                                "analyzer": "ik_smart",
                                "search_analyzer": "ik_smart",
                                "fields": {
                                    "keyword": {"type": "keyword", "ignore_above": 256}
                                }
                            },
                            "log_level": {"type": "keyword"},
                            "service_name": {"type": "keyword"},
                            "template": {
                                "type": "text",
                                "analyzer": "ik_smart",
                                "fields": {
                                    "keyword": {"type": "keyword", "ignore_above": 256}
                                }
                            },
                            "cluster_id": {"type": "keyword"},
                            "template_id": {"type": "keyword"}
                        }
                    }
                }

                # 创建索引
                self.target_es_connector.es_client.indices.create(
                    index=target_index,
                    body=settings
                )
                logger.info(f"为目标索引 {target_index} 创建了正确的IK分词器映射")

            # 创建字段映射 - 从DataFrame列名到ES标准字段名
            field_mappings = {
                self.message_field: "message",  # 源消息字段 → message
                self.level_field: "log_level",  # 源级别字段 → log_level
                self.service_field: "service_name",  # 源服务字段 → service_name
                # 时间戳字段通常保持不变
                "@timestamp": "@timestamp"
            }

            logger.info(f"保存到ES使用字段映射: {field_mappings}")

            total, success = self.target_es_connector.save_to_new_index(
                clustered_df,
                target_index=target_index,
                field_mappings=field_mappings
            )
            processing_summary["es_export"] = {
                "total": total,
                "success": success,
                "target_index": target_index,
                "field_mappings": field_mappings
            }

        return clustered_df, processing_summary


    def extract_error_patterns(
            self,
            df: pd.DataFrame,
            log_level_column: str = None,
            error_levels: List[str] = ["ERROR", "FATAL"],
            message_column: str = None,
            stack_trace_column: Optional[str] = "stack_trace"
    ) -> Dict[str, Any]:
        """
        提取错误模式

        Args:
            df: 日志DataFrame
            log_level_column: 日志级别列名（如果为None则使用配置的level_field）
            error_levels: 错误级别列表
            message_column: 消息列名（如果为None则使用配置的message_field）
            stack_trace_column: 堆栈跟踪列名

        Returns:
            错误模式摘要
        """
        # 使用配置的字段名或默认值
        log_level_column = log_level_column or self.level_field
        message_column = message_column or self.message_field

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
            target_index: Optional[str] = None,
            prefilter: bool = True,  # Add this parameter
            time_window_minutes: int = 30  # Add this parameter
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

        # 执行处理，传递新增的参数
        return self.extract_and_process(
            start_time=start_time,
            end_time=end_time,
            max_logs=max_logs,
            save_to_file=save_to_file,
            save_to_es=save_to_es,
            target_index=target_idx,
            prefilter=prefilter,  # Pass through prefilter parameter
            time_window_minutes=time_window_minutes  # Pass through time_window_minutes parameter
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
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

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