# analysis/anomaly_detector.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from utils import get_logger
from config import get_config

logger = get_logger(__name__)


class LogAnomalyDetector:
    """
    日志异常检测器，用于识别各种类型的异常

    主要功能：
    1. 时序异常检测 - 检测错误数量的时间序列异常
    2. 数量异常检测 - 检测错误数量的突然变化
    3. 新错误类型检测 - 检测以前未见过的错误类型
    4. 错误组合检测 - 检测不寻常的错误组合
    """

    def __init__(
            self,
            time_window: timedelta = timedelta(hours=1),
            sensitivity: float = 0.95,
            min_samples: int = 10,
            history_window: timedelta = timedelta(days=7),
            use_isolation_forest: bool = True,
            field_mappings: Optional[Dict[str, str]] = None
    ):
        """
        初始化异常检测器

        Args:
            time_window: 用于聚合错误的时间窗口
            sensitivity: 异常检测敏感度 (0-1)，越高越敏感
            min_samples: 进行检测所需的最小样本数
            history_window: 用于建立基线的历史窗口
            use_isolation_forest: 是否使用隔离森林算法
            field_mappings: 字段映射配置
        """
        self.config = get_config()
        self.time_window = time_window
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        self.history_window = history_window
        self.use_isolation_forest = use_isolation_forest

        # 设置字段映射
        self.field_mappings = field_mappings or {}
        self.message_field = self.field_mappings.get("message_field", "message")
        self.level_field = self.field_mappings.get("level_field", "log_level")
        self.service_field = self.field_mappings.get("service_field", "service_name")
        self.timestamp_field = self.field_mappings.get("timestamp_field", "@timestamp")

        # 历史数据
        self.error_history = {}
        self.known_error_types = set()
        self.baseline_error_rates = {}

        logger.info(f"日志异常检测器初始化，时间窗口: {time_window}, 敏感度: {sensitivity}")

    def detect_anomalies(
            self,
            df: pd.DataFrame,
            timestamp_col: str = None,
            message_col: str = None,
            level_col: str = None,
            error_levels: List[str] = ['ERROR', 'FATAL'],
            template_col: Optional[str] = 'template',
            cluster_id_col: Optional[str] = 'cluster_id',
            service_name_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        检测日志数据中的异常

        Args:
            df: 日志DataFrame
            timestamp_col: 时间戳列名（如果为None则使用配置的timestamp_field）
            message_col: 消息列名（如果为None则使用配置的message_field）
            level_col: 日志级别列名（如果为None则使用配置的level_field）
            error_levels: 错误级别列表
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）
            service_name_col: 服务名称列名（如果为None则使用配置的service_field）

        Returns:
            检测结果字典
        """
        # 使用配置的字段名或提供的参数
        timestamp_col = timestamp_col or self.timestamp_field
        message_col = message_col or self.message_field
        level_col = level_col or self.level_field
        service_name_col = service_name_col or self.service_field

        if df.empty:
            logger.warning("DataFrame为空，无法进行异常检测")
            return {"status": "no_data", "anomalies": []}

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
            logger.info("没有错误日志，跳过异常检测")
            return {"status": "no_errors", "anomalies": []}

        # 按服务名称分组处理
        if service_name_col in df.columns:
            results = self._detect_anomalies_by_service(
                df, error_df, timestamp_col, message_col, level_col,
                template_col, cluster_id_col, service_name_col
            )
        else:
            # 全局处理
            results = self._detect_all_anomalies(
                df, error_df, timestamp_col, message_col, level_col,
                template_col, cluster_id_col
            )

        return results

    def _detect_anomalies_by_service(
            self,
            df: pd.DataFrame,
            error_df: pd.DataFrame,
            timestamp_col: str,
            message_col: str,
            level_col: str,
            template_col: Optional[str],
            cluster_id_col: Optional[str],
            service_name_col: str
    ) -> Dict[str, Any]:
        """
        按服务名称分组检测异常

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
            检测结果字典
        """
        services = df[service_name_col].unique()

        all_anomalies = []
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
                    "anomalies": []
                }
                continue

            logger.info(f"检测服务 '{service}' 的异常 ({len(service_error_df)} 条错误记录)")

            # 检测此服务的异常
            service_anomalies = self._detect_all_anomalies(
                service_df, service_error_df, timestamp_col,
                message_col, level_col, template_col, cluster_id_col
            )

            # 将服务名添加到异常记录中
            for anomaly in service_anomalies.get("anomalies", []):
                anomaly["service"] = service
                all_anomalies.append(anomaly)

            service_results[service] = service_anomalies

        # 汇总结果
        return {
            "status": "success" if all_anomalies else "no_anomalies",
            "anomalies": all_anomalies,
            "service_results": service_results
        }

    def _detect_all_anomalies(
            self,
            df: pd.DataFrame,
            error_df: pd.DataFrame,
            timestamp_col: str,
            message_col: str,
            level_col: str,
            template_col: Optional[str],
            cluster_id_col: Optional[str]
    ) -> Dict[str, Any]:
        """
        检测所有类型的异常

        Args:
            df: 完整日志DataFrame
            error_df: 只包含错误的DataFrame
            timestamp_col: 时间戳列名
            message_col: 消息列名
            level_col: 日志级别列名
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）

        Returns:
            检测结果字典
        """
        anomalies = []

        # 1. 时序异常检测
        time_anomalies = self._detect_time_series_anomalies(
            error_df, timestamp_col, template_col, cluster_id_col
        )
        anomalies.extend(time_anomalies)

        # 2. 数量异常检测
        volume_anomalies = self._detect_volume_anomalies(
            error_df, timestamp_col, template_col, cluster_id_col
        )
        anomalies.extend(volume_anomalies)

        # 3. 新错误类型检测
        if template_col in error_df.columns:
            new_error_anomalies = self._detect_new_error_types(
                error_df, timestamp_col, message_col, template_col
            )
            anomalies.extend(new_error_anomalies)

        # 4. 错误组合检测
        if template_col in error_df.columns and cluster_id_col in error_df.columns:
            combo_anomalies = self._detect_error_combinations(
                error_df, timestamp_col, template_col, cluster_id_col
            )
            anomalies.extend(combo_anomalies)

        # 返回结果
        return {
            "status": "success" if anomalies else "no_anomalies",
            "anomalies": anomalies,
            "total_logs": len(df),
            "error_logs": len(error_df),
            "detection_time": datetime.now().isoformat()
        }

    def _detect_time_series_anomalies(
            self,
            error_df: pd.DataFrame,
            timestamp_col: str,
            template_col: Optional[str] = None,
            cluster_id_col: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        检测时间序列异常

        Args:
            error_df: 错误日志DataFrame
            timestamp_col: 时间戳列名
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）

        Returns:
            异常列表
        """
        if len(error_df) < self.min_samples:
            logger.info(f"样本数量不足 ({len(error_df)} < {self.min_samples})，跳过时间序列异常检测")
            return []

        anomalies = []

        try:
            # 按时间窗口聚合错误
            error_df = error_df.copy()
            error_df['time_bucket'] = error_df[timestamp_col].dt.floor(freq=self._get_freq_str())
            error_count = error_df.groupby('time_bucket').size()

            # 检查是否有足够的时间桶
            if len(error_count) < self.min_samples:
                logger.info(f"时间桶数量不足 ({len(error_count)} < {self.min_samples})，跳过时间序列异常检测")
                return []

            # 通过移动平均线检测异常
            rolling_mean = error_count.rolling(window=max(3, len(error_count) // 5)).mean()
            rolling_std = error_count.rolling(window=max(3, len(error_count) // 5)).std()

            # 处理前几个没有足够历史的点
            rolling_mean = rolling_mean.fillna(error_count.mean())
            rolling_std = rolling_std.fillna(error_count.std())

            # 设置异常阈值
            threshold = self.sensitivity * 3  # 3倍标准差，由敏感度调整

            # 计算Z分数
            z_scores = (error_count - rolling_mean) / rolling_std.clip(lower=1.0)  # 避免除以零

            # 找出异常
            anomaly_points = z_scores[z_scores.abs() > threshold]

            # 创建异常记录
            for timestamp, z_score in anomaly_points.items():
                direction = "increase" if z_score > 0 else "decrease"
                severity = min(1.0, abs(z_score) / (threshold * 2))  # 归一化严重程度

                # 获取这个时间点的详细信息
                period_errors = error_df[error_df['time_bucket'] == timestamp]

                # 获取主要错误类型
                error_types = {}
                if template_col in period_errors.columns:
                    error_types = period_errors[template_col].value_counts().head(3).to_dict()

                anomaly = {
                    "type": "time_series",
                    "timestamp": timestamp.isoformat(),
                    "severity": float(severity),
                    "description": f"错误数量{direction}异常 (Z-Score: {z_score:.2f})",
                    "details": {
                        "actual_count": int(error_count[timestamp]),
                        "expected_count": float(rolling_mean[timestamp]),
                        "z_score": float(z_score),
                        "error_types": error_types
                    }
                }

                anomalies.append(anomaly)

            logger.info(f"检测到 {len(anomalies)} 个时间序列异常")

        except Exception as e:
            logger.error(f"时间序列异常检测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        return anomalies

    def _detect_volume_anomalies(
            self,
            error_df: pd.DataFrame,
            timestamp_col: str,
            template_col: Optional[str] = None,
            cluster_id_col: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        检测错误数量异常

        Args:
            error_df: 错误日志DataFrame
            timestamp_col: 时间戳列名
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）

        Returns:
            异常列表
        """
        if len(error_df) < self.min_samples:
            return []

        anomalies = []

        try:
            # 按错误类型分组
            groupby_cols = []
            if template_col and template_col in error_df.columns:
                groupby_cols.append(template_col)
            elif cluster_id_col and cluster_id_col in error_df.columns:
                groupby_cols.append(cluster_id_col)

            if not groupby_cols:
                # 无法按错误类型分组，跳过
                return []

            # 添加时间分组
            error_df = error_df.copy()
            error_df['time_bucket'] = error_df[timestamp_col].dt.floor(freq=self._get_freq_str())
            groupby_cols.append('time_bucket')

            # 计算每个错误类型每个时间桶的数量
            error_counts = error_df.groupby(groupby_cols).size().reset_index(name='count')

            # 如果有足够的数据，使用隔离森林检测异常
            if len(error_counts) >= self.min_samples and self.use_isolation_forest:
                # 准备特征
                features = error_counts.pivot_table(
                    index='time_bucket',
                    columns=groupby_cols[0] if len(groupby_cols) > 1 else 'dummy',
                    values='count',
                    fill_value=0
                )

                # 标准化
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)

                # 使用隔离森林
                contamination = 1.0 - self.sensitivity  # 转换敏感度为污染率
                clf = IsolationForest(contamination=contamination, random_state=42)
                pred = clf.fit_predict(scaled_features)

                # 找出异常点
                anomaly_indices = features.index[pred == -1]

                for idx in anomaly_indices:
                    # 获取这个时间点的错误记录
                    time_errors = error_df[error_df['time_bucket'] == idx]

                    # 获取主要错误类型
                    error_types = {}
                    if template_col in time_errors.columns:
                        error_types = time_errors[template_col].value_counts().head(3).to_dict()

                    # 获取异常分数
                    anomaly_score = clf.score_samples([features.loc[idx].values])[0]
                    severity = min(1.0, max(0.0, -anomaly_score / 0.5))  # 归一化严重程度

                    anomaly = {
                        "type": "volume",
                        "timestamp": idx.isoformat(),
                        "severity": float(severity),
                        "description": f"错误组合数量异常 (Score: {anomaly_score:.2f})",
                        "details": {
                            "total_errors": int(time_errors.shape[0]),
                            "anomaly_score": float(anomaly_score),
                            "error_types": error_types
                        }
                    }

                    anomalies.append(anomaly)

            logger.info(f"检测到 {len(anomalies)} 个数量异常")

        except Exception as e:
            logger.error(f"数量异常检测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        return anomalies

    def _detect_new_error_types(
            self,
            error_df: pd.DataFrame,
            timestamp_col: str,
            message_col: str,
            template_col: str
    ) -> List[Dict[str, Any]]:
        """
        检测新出现的错误类型

        Args:
            error_df: 错误日志DataFrame
            timestamp_col: 时间戳列名
            message_col: 消息列名
            template_col: 模板列名

        Returns:
            异常列表
        """
        if template_col not in error_df.columns:
            return []

        anomalies = []

        try:
            # 获取错误模板
            templates = set(error_df[template_col].unique())

            # 找出新的错误类型
            new_templates = templates - self.known_error_types

            # 将新错误类型添加到已知错误类型集合
            self.known_error_types.update(templates)

            # 如果有新错误类型，创建异常记录
            for template in new_templates:
                # 获取此模板的错误记录
                template_errors = error_df[error_df[template_col] == template]
                first_occurrence = template_errors[timestamp_col].min()

                # 获取示例消息
                example_message = ""
                if message_col in template_errors.columns:
                    example_message = template_errors.iloc[0][message_col]

                anomaly = {
                    "type": "new_error",
                    "timestamp": first_occurrence.isoformat(),
                    "severity": 0.7,  # 默认严重度
                    "description": f"发现新的错误类型",
                    "details": {
                        "template": template,
                        "count": int(len(template_errors)),
                        "example": example_message
                    }
                }

                anomalies.append(anomaly)

            logger.info(f"检测到 {len(anomalies)} 个新错误类型")

        except Exception as e:
            logger.error(f"新错误类型检测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        return anomalies

    def _detect_error_combinations(
            self,
            error_df: pd.DataFrame,
            timestamp_col: str,
            template_col: str,
            cluster_id_col: str
    ) -> List[Dict[str, Any]]:
        """
        检测不寻常的错误组合

        Args:
            error_df: 错误日志DataFrame
            timestamp_col: 时间戳列名
            template_col: 模板列名
            cluster_id_col: 聚类ID列名

        Returns:
            异常列表
        """
        if template_col not in error_df.columns or cluster_id_col not in error_df.columns:
            return []

        if len(error_df) < self.min_samples:
            return []

        anomalies = []

        try:
            # 按时间窗口分组
            error_df = error_df.copy()
            error_df['time_bucket'] = error_df[timestamp_col].dt.floor(freq=self._get_freq_str())

            # 对每个时间窗口进行分析
            time_buckets = error_df['time_bucket'].unique()

            for bucket in time_buckets:
                bucket_errors = error_df[error_df['time_bucket'] == bucket]

                # 如果这个窗口的错误太少，跳过
                if len(bucket_errors) < 3:
                    continue

                # 获取错误类型和聚类分布
                template_counts = bucket_errors[template_col].value_counts()
                cluster_counts = bucket_errors[cluster_id_col].value_counts()

                # 检查是否有特殊的错误组合
                # 例如：一个窗口中有多种类型的错误，且数量接近
                if len(template_counts) >= 3:
                    top_errors = template_counts.head(3)

                    # 如果前三个错误数量接近，可能是相关的错误链
                    if top_errors.max() / top_errors.min() < 2:
                        anomaly = {
                            "type": "error_combination",
                            "timestamp": bucket.isoformat(),
                            "severity": 0.6,
                            "description": f"检测到可能的错误链",
                            "details": {
                                "error_types": top_errors.to_dict(),
                                "total_errors": int(len(bucket_errors))
                            }
                        }
                        anomalies.append(anomaly)

            logger.info(f"检测到 {len(anomalies)} 个错误组合异常")

        except Exception as e:
            logger.error(f"错误组合检测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        return anomalies

    def _get_freq_str(self) -> str:
        """
        获取pandas时间频率字符串

        Returns:
            频率字符串
        """
        # 将timedelta转换为pandas频率字符串
        total_seconds = self.time_window.total_seconds()

        if total_seconds <= 60:
            return f"{int(total_seconds)}S"  # 秒
        elif total_seconds <= 3600:
            return f"{int(total_seconds // 60)}min"  # 分钟
        elif total_seconds <= 86400:
            return f"{int(total_seconds // 3600)}H"  # 小时
        else:
            return f"{int(total_seconds // 86400)}D"  # 天

    def visualize_time_series(
            self,
            error_df: pd.DataFrame,
            timestamp_col: str,
            template_col: Optional[str] = None,
            cluster_id_col: Optional[str] = None,
            figsize: Tuple[int, int] = (12, 6)
    ) -> str:
        """
        可视化错误时间序列

        Args:
            error_df: 错误日志DataFrame
            timestamp_col: 时间戳列名
            template_col: 模板列名（可选）
            cluster_id_col: 聚类ID列名（可选）
            figsize: 图像大小

        Returns:
            Base64编码的PNG图像
        """
        if error_df.empty:
            return ""

        try:
            # 按时间窗口聚合错误
            error_df = error_df.copy()
            error_df['time_bucket'] = error_df[timestamp_col].dt.floor(freq=self._get_freq_str())

            # 如果有模板或聚类ID，按此分组
            if template_col and template_col in error_df.columns:
                # 获取前5个最常见的错误类型
                top_templates = error_df[template_col].value_counts().head(5).index

                # 创建图表
                plt.figure(figsize=figsize)

                for template in top_templates:
                    # 获取此模板的时间序列
                    template_errors = error_df[error_df[template_col] == template]
                    template_counts = template_errors.groupby('time_bucket').size()

                    # 绘制此模板的时间序列
                    plt.plot(template_counts.index, template_counts.values, label=f"{str(template)[:30]}...")

                # 添加总体错误时间序列
                total_counts = error_df.groupby('time_bucket').size()
                plt.plot(total_counts.index, total_counts.values, 'k--', label='所有错误')

            else:
                # 只按时间聚合
                error_counts = error_df.groupby('time_bucket').size()

                # 创建图表
                plt.figure(figsize=figsize)
                plt.plot(error_counts.index, error_counts.values)

            # 设置图表
            plt.title('错误时间序列')
            plt.xlabel('时间')
            plt.ylabel('错误数量')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            if template_col and template_col in error_df.columns:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()

            # 将图表保存为内存中的图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            buf.seek(0)

            # 返回Base64编码的图像
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        except Exception as e:
            logger.error(f"可视化时间序列失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""