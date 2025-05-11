# deduplication.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LogDeduplicator:
    """通用智能日志去重系统"""

    def __init__(
            self,
            similarity_threshold: float = 0.92,
            time_window_hours: int = 1,
            min_cluster_size: int = 2,
            novelty_threshold: float = 0.05,
            keep_latest_count: int = 3,
            keep_anomalies: bool = True,
            max_memory_logs: int = 10000
    ):
        """
        初始化日志去重器

        Args:
            similarity_threshold: 日志相似度阈值 (0-1)，越高表示越严格
            time_window_hours: 时间窗口（小时），用于限制最近日志比较范围
            min_cluster_size: 形成聚类所需的最小日志数量
            novelty_threshold: 语义新颖性阈值，高于此值的日志被视为有新信息
            keep_latest_count: 每个聚类保留的最新日志数
            keep_anomalies: 是否保留所有异常日志
            max_memory_logs: 内存中保存的最大历史日志数
        """
        self.similarity_threshold = similarity_threshold
        self.time_window_hours = time_window_hours
        self.min_cluster_size = min_cluster_size
        self.novelty_threshold = novelty_threshold
        self.keep_latest_count = keep_latest_count
        self.keep_anomalies = keep_anomalies
        self.max_memory_logs = max_memory_logs

        # 历史日志缓存
        self.recent_logs = []
        self.recent_vectors = []
        self.recent_timestamps = []

        # 性能统计
        self.total_processed = 0
        self.total_deduplicated = 0
        self.last_reset = datetime.now()

    def semantic_novelty_filter(self, new_vector: List[float]) -> bool:
        """
        检查日志是否有足够的语义新颖性

        Args:
            new_vector: 新日志的向量表示

        Returns:
            是否保留该日志
        """
        if not self.recent_vectors:
            return True  # 没有历史日志，保留

        # 计算与历史日志的最小语义距离
        distances = [cosine(new_vector, vec) for vec in self.recent_vectors]
        min_distance = min(distances)

        # 如果差异足够大，保留该日志
        return min_distance > self.novelty_threshold

    def update_recent_logs(self, log_id: str, vector: List[float], timestamp: datetime):
        """
        更新最近日志缓存

        Args:
            log_id: 日志ID
            vector: 日志向量
            timestamp: 日志时间戳
        """
        self.recent_logs.append(log_id)
        self.recent_vectors.append(vector)
        self.recent_timestamps.append(timestamp)

        # 如果缓存超过最大大小，移除最旧的日志
        if len(self.recent_logs) > self.max_memory_logs:
            self.recent_logs.pop(0)
            self.recent_vectors.pop(0)
            self.recent_timestamps.pop(0)

        # 定期清理旧日志
        current_time = datetime.now()
        if (current_time - self.last_reset).total_seconds() > 3600:  # 每小时
            self._cleanup_old_logs()
            self.last_reset = current_time

    def _cleanup_old_logs(self):
        """清理时间窗口外的旧日志"""
        if not self.recent_timestamps:
            return

        cutoff_time = datetime.now() - timedelta(hours=self.time_window_hours)
        valid_indices = [i for i, ts in enumerate(self.recent_timestamps)
                         if ts >= cutoff_time]

        # 更新缓存
        self.recent_logs = [self.recent_logs[i] for i in valid_indices]
        self.recent_vectors = [self.recent_vectors[i] for i in valid_indices]
        self.recent_timestamps = [self.recent_timestamps[i] for i in valid_indices]

        logger.info(f"清理旧日志缓存，当前保留 {len(self.recent_logs)} 条日志")

    def smart_cluster_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用DBSCAN聚类进行智能去重

        Args:
            df: 日志DataFrame，必须包含向量列

        Returns:
            去重后的DataFrame
        """
        if df.empty:
            return df

        start_time = time.time()
        original_count = len(df)

        # 提取向量
        vectors = np.array(df['vector'].tolist())
        if len(vectors) < self.min_cluster_size:
            return df  # 日志数量太少，不进行聚类

        # 聚类参数换算 - 相似度阈值转换为距离阈值
        eps = 1.0 - self.similarity_threshold

        # 执行DBSCAN聚类
        clustering = DBSCAN(
            eps=eps,
            min_samples=self.min_cluster_size,
            metric='cosine',
            n_jobs=-1  # 使用所有CPU
        ).fit(vectors)

        # 将聚类ID添加到DataFrame
        df['cluster_id'] = clustering.labels_

        # 保留重要日志
        representative_logs = []

        # 噪声点 (未被聚类的点) 全部保留
        noise_logs = df[df['cluster_id'] == -1]
        if not noise_logs.empty:
            representative_logs.append(noise_logs)

        # 处理每个聚类
        for cluster_id in sorted(set(df['cluster_id'])):
            if cluster_id == -1:
                continue  # 噪声点已处理

            cluster_df = df[df['cluster_id'] == cluster_id]

            # 策略1: 保留每个聚类中最新的N条日志
            if '@timestamp' in cluster_df.columns:
                sorted_df = cluster_df.sort_values('@timestamp', ascending=False)
                representative_logs.append(sorted_df.head(self.keep_latest_count))
            else:
                # 如果没有时间戳，随机保留N条
                representative_logs.append(cluster_df.sample(
                    min(self.keep_latest_count, len(cluster_df))
                ))

            # 策略2: 如果启用，保留所有异常日志
            if self.keep_anomalies and 'prediction' in cluster_df.columns:
                anomalies = cluster_df[cluster_df['prediction'] == 1]
                if not anomalies.empty:
                    # 避免重复添加已保留的异常日志
                    already_kept = representative_logs[-1].index
                    new_anomalies = anomalies[~anomalies.index.isin(already_kept)]
                    if not new_anomalies.empty:
                        representative_logs.append(new_anomalies)

        # 合并所有保留的日志
        result_df = pd.concat(representative_logs) if representative_logs else df.head(0)

        # 更新统计
        deduplicated_count = original_count - len(result_df)
        self.total_processed += original_count
        self.total_deduplicated += deduplicated_count

        processing_time = time.time() - start_time
        logger.info(f"智能聚类去重: 处理 {original_count} 条日志，移除 {deduplicated_count} 条"
                    f" ({deduplicated_count / original_count * 100:.1f}%)，耗时 {processing_time:.2f}秒")

        return result_df

    def process_single_log(self, log_id: str, vector: List[float],
                           timestamp: datetime, is_anomaly: bool = False) -> bool:
        """
        处理单条日志，判断是否应保留

        Args:
            log_id: 日志ID
            vector: 日志向量表示
            timestamp: 日志时间戳
            is_anomaly: 是否为异常日志

        Returns:
            是否应保留该日志
        """
        # 异常日志始终保留
        if is_anomaly and self.keep_anomalies:
            self.update_recent_logs(log_id, vector, timestamp)
            return True

        # 检查语义新颖性
        if self.semantic_novelty_filter(vector):
            self.update_recent_logs(log_id, vector, timestamp)
            return True

        # 日志被去重
        self.total_processed += 1
        self.total_deduplicated += 1
        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        获取去重统计信息

        Returns:
            统计信息字典
        """
        dedup_rate = 0
        if self.total_processed > 0:
            dedup_rate = self.total_deduplicated / self.total_processed

        return {
            "total_processed": self.total_processed,
            "total_deduplicated": self.total_deduplicated,
            "deduplication_rate": dedup_rate,
            "recent_logs_count": len(self.recent_logs),
            "current_memory_usage_mb": get_size_mb(self),
            "settings": {
                "similarity_threshold": self.similarity_threshold,
                "novelty_threshold": self.novelty_threshold,
                "time_window_hours": self.time_window_hours
            }
        }

    def reset_stats(self):
        """重置统计数据"""
        self.total_processed = 0
        self.total_deduplicated = 0
        self.last_reset = datetime.now()


def get_size_mb(obj) -> float:
    """估算对象内存使用量 (MB)"""
    import sys
    return sys.getsizeof(obj) / (1024 * 1024)