# analysis/pattern_analyzer.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import io
import base64
from collections import Counter, defaultdict
import re

from utils import get_logger, extract_keywords, clean_text
from config import get_config

logger = get_logger(__name__)


class LogPatternAnalyzer:
    """
    日志模式分析器，用于发现日志数据中的模式和关系

    主要功能：
    1. 错误链分析 - 发现错误之间的因果关系
    2. 时间模式分析 - 分析错误发生的时间模式
    3. 共现分析 - 分析同时出现的错误模式
    4. 服务依赖分析 - 分析不同服务之间的依赖关系
    """

    def __init__(
            self,
            time_window: timedelta = timedelta(minutes=5),
            min_confidence: float = 0.6,
            min_support: int = 3,
            max_patterns: int = 20
    ):
        """
        初始化模式分析器

        Args:
            time_window: 考虑事件关联的时间窗口
            min_confidence: 最小置信度，用于关联规则
            min_support: 最小支持度，模式必须出现的最小次数
            max_patterns: 返回的最大模式数量
        """
        self.config = get_config()
        self.time_window = time_window
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_patterns = max_patterns

        logger.info(f"日志模式分析器初始化，时间窗口: {time_window}, 最小置信度: {min_confidence}")

    def analyze_patterns(
            self,
            df: pd.DataFrame,
            timestamp_col: str = '@timestamp',
            message_col: str = 'message',
            level_col: str = 'log_level',
            error_levels: List[str] = ['ERROR', 'FATAL'],
            template_col: Optional[str] = 'template',
            cluster_id_col: Optional[str] = 'cluster_id',
            service_name_col: Optional[str] = 'service_name'
    ) -> Dict[str, Any]:
        """
        分析日志数据中的模式

        Args:
            df: 日志DataFrame
            timestamp_col: 时间戳列名
            message_col: 消息列名
            level_col: 日志级别列名
            error_levels: 错误级别列表
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）
            service_name_col: 服务名称列名（可选）

        Returns:
            模式分析结果字典
        """
        if df.empty:
            logger.warning("DataFrame为空，无法进行模式分析")
            return {"status": "no_data", "patterns": []}

        # 确保时间戳列存在
        if timestamp_col not in df.columns:
            logger.error(f"时间戳列 '{timestamp_col}' 不在DataFrame中")
            return {"status": "error", "message": f"缺少时间戳列: {timestamp_col}"}

        # 将时间戳转换为datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except Exception as e:
                logger.error(f"转换时间戳失败: {str(e)}")
                return {"status": "error", "message": f"时间戳转换失败: {str(e)}"}

        # 过滤出错误日志
        error_df = df
        if level_col in df.columns:
            error_df = df[df[level_col].isin(error_levels)]

        if error_df.empty:
            logger.info("没有错误日志，跳过模式分析")
            return {"status": "no_errors", "patterns": []}

        # 按服务名称分组处理
        results = {}

        if service_name_col in df.columns:
            # 多服务分析
            service_patterns = self._analyze_patterns_by_service(
                df, error_df, timestamp_col, message_col, level_col,
                template_col, cluster_id_col, service_name_col
            )
            results["service_patterns"] = service_patterns

            # 服务间依赖分析
            service_dependencies = self._analyze_service_dependencies(
                df, timestamp_col, service_name_col, error_levels, level_col
            )
            results["service_dependencies"] = service_dependencies

        # 错误链分析
        error_chains = self._analyze_error_chains(
            error_df, timestamp_col, message_col, template_col, cluster_id_col
        )
        results["error_chains"] = error_chains

        # 时间模式分析
        time_patterns = self._analyze_time_patterns(
            error_df, timestamp_col, template_col, cluster_id_col
        )
        results["time_patterns"] = time_patterns

        # 计算模式总数
        total_patterns = (
                len(error_chains.get("patterns", [])) +
                len(time_patterns.get("patterns", []))
        )

        if "service_patterns" in results:
            for service, patterns in results["service_patterns"].items():
                total_patterns += len(patterns.get("patterns", []))

        # 返回结果
        return {
            "status": "success" if total_patterns > 0 else "no_patterns",
            "total_patterns": total_patterns,
            **results
        }

    def _analyze_patterns_by_service(
            self,
            df: pd.DataFrame,
            error_df: pd.DataFrame,
            timestamp_col: str,
            message_col: str,
            level_col: str,
            template_col: Optional[str],
            cluster_id_col: Optional[str],
            service_name_col: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        按服务名称分组分析模式

        Args:
            df: 完整日志DataFrame
            error_df: 只包含错误的DataFrame
            timestamp_col: 时间戳列名
            message_col: 消息列名
            level_col: 日志级别列名
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）
            service_name_col: 服务名称列名

        Returns:
            按服务分组的模式分析结果
        """
        services = df[service_name_col].unique()

        service_results = {}

        for service in services:
            if pd.isna(service):
                # 跳过空服务名
                continue

            service_df = df[df[service_name_col] == service]
            service_error_df = error_df[error_df[service_name_col] == service]

            if service_error_df.empty:
                service_results[service] = {
                    "status": "no_errors",
                    "patterns": []
                }
                continue

            logger.info(f"分析服务 '{service}' 的模式 ({len(service_error_df)} 条错误记录)")

            # 错误链分析
            error_chains = self._analyze_error_chains(
                service_error_df, timestamp_col, message_col, template_col, cluster_id_col
            )

            # 时间模式分析
            time_patterns = self._analyze_time_patterns(
                service_error_df, timestamp_col, template_col, cluster_id_col
            )

            service_results[service] = {
                "status": "success" if error_chains["patterns"] or time_patterns["patterns"] else "no_patterns",
                "patterns": [
                    *error_chains["patterns"],
                    *time_patterns["patterns"]
                ],
                "total_logs": len(service_df),
                "error_logs": len(service_error_df)
            }

        return service_results

    def _analyze_error_chains(
            self,
            error_df: pd.DataFrame,
            timestamp_col: str,
            message_col: str,
            template_col: Optional[str] = None,
            cluster_id_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析错误链 - 一个错误导致另一个错误的模式

        Args:
            error_df: 错误日志DataFrame
            timestamp_col: 时间戳列名
            message_col: 消息列名
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）

        Returns:
            错误链分析结果
        """
        if len(error_df) < self.min_support:
            return {"status": "insufficient_data", "patterns": []}

        try:
            # 确定错误标识符
            error_id_col = None
            if template_col and template_col in error_df.columns:
                error_id_col = template_col
            elif cluster_id_col and cluster_id_col in error_df.columns:
                error_id_col = cluster_id_col
            else:
                # 无法识别错误类型，尝试从消息中提取关键词
                error_df = error_df.copy()
                error_df['error_keywords'] = error_df[message_col].apply(
                    lambda x: '_'.join(extract_keywords(x, max_keywords=3))
                )
                error_id_col = 'error_keywords'

            # 按时间排序
            error_df = error_df.sort_values(timestamp_col)

            # 构建错误序列
            error_sequences = []
            current_sequence = []
            last_timestamp = None

            for _, row in error_df.iterrows():
                current_timestamp = row[timestamp_col]
                current_error = row[error_id_col]

                # 如果是新序列
                if last_timestamp is None or (current_timestamp - last_timestamp) > self.time_window:
                    if current_sequence:
                        error_sequences.append(current_sequence)
                    current_sequence = [current_error]
                else:
                    # 避免连续相同错误
                    if not current_sequence or current_error != current_sequence[-1]:
                        current_sequence.append(current_error)

                last_timestamp = current_timestamp

            # 添加最后一个序列
            if current_sequence:
                error_sequences.append(current_sequence)

            # 如果序列太少，返回空结果
            if len(error_sequences) < self.min_support:
                return {"status": "insufficient_sequences", "patterns": []}

            # 寻找频繁错误对
            error_pairs = defaultdict(int)

            for sequence in error_sequences:
                # 只考虑长度大于1的序列
                if len(sequence) < 2:
                    continue

                # 检查每个相邻错误对
                for i in range(len(sequence) - 1):
                    error_pair = (sequence[i], sequence[i + 1])
                    error_pairs[error_pair] += 1

            # 过滤支持度低的错误对
            frequent_pairs = {
                pair: count for pair, count in error_pairs.items()
                if count >= self.min_support
            }

            # 计算置信度
            error_counts = Counter([error for sequence in error_sequences for error in sequence])

            patterns = []

            for (error1, error2), count in frequent_pairs.items():
                confidence = count / error_counts[error1]

                if confidence >= self.min_confidence:
                    pattern = {
                        "type": "error_chain",
                        "description": f"错误链模式: {error1} → {error2}",
                        "confidence": float(confidence),
                        "support": int(count),
                        "details": {
                            "source_error": str(error1),
                            "target_error": str(error2),
                            "source_count": int(error_counts[error1]),
                            "target_count": int(error_counts[error2])
                        }
                    }
                    patterns.append(pattern)

            # 按置信度排序
            patterns.sort(key=lambda x: (x["confidence"], x["support"]), reverse=True)

            # 限制返回的模式数量
            patterns = patterns[:self.max_patterns]

            # 构建错误图
            graph_image = ""
            if patterns:
                graph_image = self._generate_error_chain_graph(patterns, error_counts)

            return {
                "status": "success" if patterns else "no_patterns",
                "patterns": patterns,
                "total_sequences": len(error_sequences),
                "graph_image": graph_image
            }

        except Exception as e:
            logger.error(f"错误链分析失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "patterns": []}

    def _analyze_time_patterns(
            self,
            error_df: pd.DataFrame,
            timestamp_col: str,
            template_col: Optional[str] = None,
            cluster_id_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析时间模式 - 错误在特定时间点或周期出现的模式

        Args:
            error_df: 错误日志DataFrame
            timestamp_col: 时间戳列名
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）

        Returns:
            时间模式分析结果
        """
        if len(error_df) < self.min_support:
            return {"status": "insufficient_data", "patterns": []}

        try:
            # 提取时间特征
            error_df = error_df.copy()

            # 提取小时、星期几、是否工作时间等特征
            error_df['hour'] = error_df[timestamp_col].dt.hour
            error_df['day_of_week'] = error_df[timestamp_col].dt.dayofweek  # 0=Monday, 6=Sunday
            error_df['is_weekend'] = error_df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
            error_df['is_business_hours'] = (
                    (error_df['hour'] >= 9) & (error_df['hour'] < 18) &
                    (~error_df['is_weekend'])
            )

            patterns = []

            # 确定错误标识符
            error_id_col = None
            if template_col and template_col in error_df.columns:
                error_id_col = template_col
            elif cluster_id_col and cluster_id_col in error_df.columns:
                error_id_col = cluster_id_col

            # 如果有错误标识符，按错误类型分组分析
            if error_id_col:
                error_types = error_df[error_id_col].value_counts()

                # 只分析出现频率高的错误类型
                for error_type, count in error_types.items():
                    if count < self.min_support:
                        continue

                    type_df = error_df[error_df[error_id_col] == error_type]

                    # 检查小时分布
                    hour_counts = type_df['hour'].value_counts()
                    total_hours = hour_counts.sum()

                    for hour, hour_count in hour_counts.items():
                        hour_ratio = hour_count / total_hours

                        # 如果错误集中在某个小时
                        if hour_ratio >= 0.3 and hour_count >= self.min_support:
                            pattern = {
                                "type": "time_pattern",
                                "description": f"错误 '{error_type}' 在 {hour}:00 时集中出现",
                                "confidence": float(hour_ratio),
                                "support": int(hour_count),
                                "details": {
                                    "error_type": str(error_type),
                                    "hour": int(hour),
                                    "count": int(hour_count),
                                    "total_count": int(count)
                                }
                            }
                            patterns.append(pattern)

                    # 检查工作时间与非工作时间
                    business_count = type_df['is_business_hours'].sum()
                    nonbusiness_count = count - business_count

                    # 如果错误主要在非工作时间发生
                    if nonbusiness_count >= self.min_support and nonbusiness_count / count >= 0.7:
                        pattern = {
                            "type": "time_pattern",
                            "description": f"错误 '{error_type}' 主要在非工作时间发生",
                            "confidence": float(nonbusiness_count / count),
                            "support": int(nonbusiness_count),
                            "details": {
                                "error_type": str(error_type),
                                "period": "非工作时间",
                                "count": int(nonbusiness_count),
                                "total_count": int(count)
                            }
                        }
                        patterns.append(pattern)

                    # 如果错误主要在工作时间发生
                    elif business_count >= self.min_support and business_count / count >= 0.7:
                        pattern = {
                            "type": "time_pattern",
                            "description": f"错误 '{error_type}' 主要在工作时间发生",
                            "confidence": float(business_count / count),
                            "support": int(business_count),
                            "details": {
                                "error_type": str(error_type),
                                "period": "工作时间",
                                "count": int(business_count),
                                "total_count": int(count)
                            }
                        }
                        patterns.append(pattern)

                    # 检查周末与工作日
                    weekend_count = type_df['is_weekend'].sum()
                    weekday_count = count - weekend_count

                    # 考虑到周末只有2天，工作日有5天，调整权重
                    weekend_ratio = weekend_count / count
                    weekday_ratio = weekday_count / count

                    adjusted_weekend_ratio = weekend_ratio * (7 / 2)  # 调整为如果均匀分布

                    # 如果错误在周末集中
                    if weekend_count >= self.min_support and adjusted_weekend_ratio >= 1.5:
                        pattern = {
                            "type": "time_pattern",
                            "description": f"错误 '{error_type}' 在周末集中出现",
                            "confidence": float(weekend_ratio),
                            "support": int(weekend_count),
                            "details": {
                                "error_type": str(error_type),
                                "period": "周末",
                                "count": int(weekend_count),
                                "total_count": int(count)
                            }
                        }
                        patterns.append(pattern)

            # 全局时间模式分析

            # 按天分组，查看错误是否集中在某几天
            error_df['date'] = error_df[timestamp_col].dt.date
            daily_counts = error_df.groupby('date').size()

            # 检查日变化模式
            if len(daily_counts) >= 3:  # 至少有3天数据
                mean_count = daily_counts.mean()
                std_count = daily_counts.std()

                # 查找异常高的日子
                high_days = daily_counts[daily_counts > mean_count + 2 * std_count]

                for day, count in high_days.items():
                    # 查看这一天的错误分布
                    day_df = error_df[error_df['date'] == day]

                    if error_id_col:
                        top_errors = day_df[error_id_col].value_counts().head(3)
                        error_details = {str(k): int(v) for k, v in top_errors.items()}
                    else:
                        error_details = {"total": int(count)}

                    pattern = {
                        "type": "time_pattern",
                        "description": f"在 {day} 错误数量异常高 ({count} 错误)",
                        "confidence": float((count - mean_count) / (std_count + 1)),  # 避免除零
                        "support": int(count),
                        "details": {
                            "date": str(day),
                            "count": int(count),
                            "average_count": float(mean_count),
                            "errors": error_details
                        }
                    }
                    patterns.append(pattern)

            # 按置信度排序
            patterns.sort(key=lambda x: (x["confidence"], x["support"]), reverse=True)

            # 限制返回的模式数量
            patterns = patterns[:self.max_patterns]

            # 生成时间热力图
            time_heatmap = ""
            if len(error_df) >= self.min_support * 3:
                time_heatmap = self._generate_time_heatmap(error_df)

            return {
                "status": "success" if patterns else "no_patterns",
                "patterns": patterns,
                "time_heatmap": time_heatmap
            }

        except Exception as e:
            logger.error(f"时间模式分析失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "patterns": []}

    def _analyze_service_dependencies(
            self,
            df: pd.DataFrame,
            timestamp_col: str,
            service_name_col: str,
            error_levels: List[str],
            level_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析服务间依赖关系

        Args:
            df: 日志DataFrame
            timestamp_col: 时间戳列名
            service_name_col: 服务名称列名
            error_levels: 错误级别列表
            level_col: 日志级别列名（可选）

        Returns:
            服务依赖分析结果
        """
        if service_name_col not in df.columns:
            return {"status": "missing_service_column", "dependencies": []}

        if len(df) < self.min_support * 5:  # 需要更多数据来分析服务依赖
            return {"status": "insufficient_data", "dependencies": []}

        try:
            # 按时间排序
            df = df.sort_values(timestamp_col)

            # 获取所有服务
            services = df[service_name_col].dropna().unique()

            if len(services) < 2:
                return {"status": "single_service", "dependencies": []}

            # 按时间窗口创建服务事件序列
            df['time_bucket'] = df[timestamp_col].dt.floor(freq=f"{int(self.time_window.total_seconds())}S")

            # 创建服务依赖矩阵
            service_matrix = np.zeros((len(services), len(services)))
            service_indices = {service: i for i, service in enumerate(services)}

            # 按时间窗口分组
            time_groups = df.groupby('time_bucket')

            for _, group in time_groups:
                # 获取此时间窗口内的服务
                window_services = group[service_name_col].dropna().unique()

                # 如果只有一个服务，跳过
                if len(window_services) < 2:
                    continue

                # 检查是否有错误
                has_error = False
                if level_col in group.columns:
                    has_error = any(group[level_col].isin(error_levels))

                # 计算共现频率
                for i, service1 in enumerate(window_services):
                    for service2 in window_services[i + 1:]:
                        idx1 = service_indices[service1]
                        idx2 = service_indices[service2]

                        # 增加计数（如果有错误，给更高权重）
                        weight = 2 if has_error else 1
                        service_matrix[idx1, idx2] += weight
                        service_matrix[idx2, idx1] += weight

            # 过滤低频共现
            min_occurrences = self.min_support
            service_matrix[service_matrix < min_occurrences] = 0

            # 构建依赖关系
            dependencies = []

            for i in range(len(services)):
                for j in range(i + 1, len(services)):
                    if service_matrix[i, j] >= min_occurrences:
                        service1 = services[i]
                        service2 = services[j]
                        strength = float(service_matrix[i, j])

                        dependency = {
                            "source": str(service1),
                            "target": str(service2),
                            "strength": strength,
                            "description": f"服务 '{service1}' 和 '{service2}' 经常同时出现"
                        }
                        dependencies.append(dependency)

            # 按强度排序
            dependencies.sort(key=lambda x: x["strength"], reverse=True)

            # 限制返回的依赖数量
            dependencies = dependencies[:self.max_patterns]

            # 生成服务依赖图
            dependency_graph = ""
            if dependencies:
                dependency_graph = self._generate_service_dependency_graph(dependencies, services)

            return {
                "status": "success" if dependencies else "no_dependencies",
                "dependencies": dependencies,
                "services": [str(s) for s in services],
                "dependency_graph": dependency_graph
            }

        except Exception as e:
            logger.error(f"服务依赖分析失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e), "dependencies": []}

    def _generate_error_chain_graph(
            self,
            patterns: List[Dict[str, Any]],
            error_counts: Dict[str, int]
    ) -> str:
        """
        生成错误链图

        Args:
            patterns: 错误链模式列表
            error_counts: 错误计数字典

        Returns:
            Base64编码的PNG图像
        """
        try:
            # 创建有向图
            G = nx.DiGraph()

            # 添加节点和边
            for pattern in patterns:
                source = pattern["details"]["source_error"]
                target = pattern["details"]["target_error"]
                weight = pattern["confidence"]

                # 添加节点（如果不存在）
                if source not in G:
                    G.add_node(source, count=error_counts.get(source, 0))
                if target not in G:
                    G.add_node(target, count=error_counts.get(target, 0))

                # 添加边
                G.add_edge(source, target, weight=weight, confidence=weight)

            # 创建图
            plt.figure(figsize=(12, 8))

            # 计算节点大小（基于错误计数）
            node_sizes = [G.nodes[n].get('count', 1) * 100 for n in G.nodes()]

            # 计算边宽度（基于置信度）
            edge_widths = [G.edges[e]['weight'] * 3 for e in G.edges()]

            # 使用spring布局
            pos = nx.spring_layout(G, seed=42)

            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)

            # 绘制边
            nx.draw_networkx_edges(
                G, pos, width=edge_widths, alpha=0.7,
                edge_color='black', arrowsize=20
            )

            # 绘制标签
            # 对长标签进行截断
            labels = {n: str(n)[:20] + ('...' if len(str(n)) > 20 else '') for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

            # 设置图表
            plt.title('错误链分析')
            plt.axis('off')
            plt.tight_layout()

            # 将图表保存为内存中的图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            buf.seek(0)

            # 返回Base64编码的图像
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        except Exception as e:
            logger.error(f"生成错误链图失败: {str(e)}")
            return ""

        def _generate_time_heatmap(self, error_df: pd.DataFrame) -> str:
            """
            生成时间热力图

            Args:
                error_df: 错误日志DataFrame

            Returns:
                Base64编码的PNG图像
            """
            try:
                # 按小时和星期几分组
                hour_day_counts = pd.crosstab(
                    error_df['hour'],
                    error_df['day_of_week'],
                    rownames=['Hour'],
                    colnames=['Day of Week']
                )

                # 重命名列
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                hour_day_counts.columns = [day_names[i] for i in hour_day_counts.columns]

                # 创建热力图
                plt.figure(figsize=(10, 8))
                plt.imshow(hour_day_counts, cmap='YlOrRd', aspect='auto')

                # 设置坐标轴
                plt.xticks(range(len(hour_day_counts.columns)), hour_day_counts.columns, rotation=45)
                plt.yticks(range(len(hour_day_counts.index)), hour_day_counts.index)

                # 添加数值标签
                for i in range(len(hour_day_counts.index)):
                    for j in range(len(hour_day_counts.columns)):
                        if not pd.isna(hour_day_counts.iloc[i, j]) and hour_day_counts.iloc[i, j] > 0:
                            plt.text(j, i, str(int(hour_day_counts.iloc[i, j])),
                                     ha='center', va='center',
                                     color='black' if hour_day_counts.iloc[
                                                          i, j] < hour_day_counts.values.max() / 2 else 'white')

                # 添加颜色条和标题
                plt.colorbar(label='Error Count')
                plt.title('Error Occurrence by Hour and Day of Week')
                plt.xlabel('Day of Week')
                plt.ylabel('Hour of Day')

                plt.tight_layout()

                # 将图表保存为内存中的图像
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close()
                buf.seek(0)

                # 返回Base64编码的图像
                return base64.b64encode(buf.getvalue()).decode('utf-8')

            except Exception as e:
                logger.error(f"生成时间热力图失败: {str(e)}")
                return ""

        def _generate_service_dependency_graph(
                self,
                dependencies: List[Dict[str, Any]],
                services: List[str]
        ) -> str:
            """
            生成服务依赖图

            Args:
                dependencies: 服务依赖列表
                services: 服务列表

            Returns:
                Base64编码的PNG图像
            """
            try:
                # 创建无向图
                G = nx.Graph()

                # 添加所有服务作为节点
                for service in services:
                    G.add_node(service)

                # 添加依赖关系作为边
                for dep in dependencies:
                    source = dep["source"]
                    target = dep["target"]
                    strength = dep["strength"]

                    G.add_edge(source, target, weight=strength)

                # 创建图
                plt.figure(figsize=(12, 10))

                # 使用spring布局
                pos = nx.spring_layout(G, seed=42, k=0.3)

                # 计算节点大小（基于连接数）
                node_degrees = dict(G.degree())
                node_sizes = [node_degrees[n] * 300 for n in G.nodes()]

                # 计算边宽度（基于强度）
                edge_widths = [G[u][v]['weight'] / 5 for u, v in G.edges()]

                # 绘制节点，使用不同颜色
                colors = plt.cm.tab20(np.linspace(0, 1, len(G.nodes())))
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.8)

                # 绘制边
                nx.draw_networkx_edges(
                    G, pos, width=edge_widths, alpha=0.5,
                    edge_color='gray'
                )

                # 绘制标签
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

                # 设置图表
                plt.title('Service Dependency Graph')
                plt.axis('off')
                plt.tight_layout()

                # 将图表保存为内存中的图像
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close()
                buf.seek(0)

                # 返回Base64编码的图像
                return base64.b64encode(buf.getvalue()).decode('utf-8')

            except Exception as e:
                logger.error(f"生成服务依赖图失败: {str(e)}")
                return ""

        def find_related_errors(
                self,
                error_df: pd.DataFrame,
                target_error: str,
                timestamp_col: str = '@timestamp',
                template_col: str = 'template',
                time_range: Optional[timedelta] = None
        ) -> List[Dict[str, Any]]:
            """
            查找与目标错误相关的其他错误

            Args:
                error_df: 错误日志DataFrame
                target_error: 目标错误
                timestamp_col: 时间戳列名
                template_col: 模板列名
                time_range: 时间范围，默认为类初始化的time_window

            Returns:
                相关错误列表
            """
            if template_col not in error_df.columns:
                return []

            if target_error not in error_df[template_col].values:
                return []

            time_range = time_range or self.time_window

            try:
                # 查找目标错误发生的时间
                target_times = error_df[error_df[template_col] == target_error][timestamp_col]

                related_errors = []

                for target_time in target_times:
                    # 查找时间窗口内的其他错误
                    time_window_start = target_time - time_range
                    time_window_end = target_time + time_range

                    window_errors = error_df[
                        (error_df[timestamp_col] >= time_window_start) &
                        (error_df[timestamp_col] <= time_window_end) &
                        (error_df[template_col] != target_error)
                        ]

                    # 统计错误频率
                    error_counts = window_errors[template_col].value_counts()

                    for error, count in error_counts.items():
                        # 检查是在目标错误之前还是之后
                        before_errors = window_errors[
                            (window_errors[template_col] == error) &
                            (window_errors[timestamp_col] < target_time)
                            ]

                        after_errors = window_errors[
                            (window_errors[template_col] == error) &
                            (window_errors[timestamp_col] > target_time)
                            ]

                        # 确定关系类型
                        relation_type = "co-occurrence"
                        if len(before_errors) > len(after_errors) * 2:
                            relation_type = "precedes"  # 主要出现在目标错误之前
                        elif len(after_errors) > len(before_errors) * 2:
                            relation_type = "follows"  # 主要出现在目标错误之后

                        related_error = {
                            "error": str(error),
                            "relation_type": relation_type,
                            "count": int(count),
                            "before_count": int(len(before_errors)),
                            "after_count": int(len(after_errors))
                        }

                        # 检查是否已存在
                        existing = next((e for e in related_errors if e["error"] == error), None)

                        if existing:
                            # 更新计数
                            existing["count"] += count
                            existing["before_count"] += len(before_errors)
                            existing["after_count"] += len(after_errors)

                            # 重新确定关系类型
                            if existing["before_count"] > existing["after_count"] * 2:
                                existing["relation_type"] = "precedes"
                            elif existing["after_count"] > existing["before_count"] * 2:
                                existing["relation_type"] = "follows"
                            else:
                                existing["relation_type"] = "co-occurrence"
                        else:
                            related_errors.append(related_error)

                # 按计数排序
                related_errors.sort(key=lambda x: x["count"], reverse=True)

                return related_errors

            except Exception as e:
                logger.error(f"查找相关错误失败: {str(e)}")
                return []