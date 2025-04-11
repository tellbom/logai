# models/clustering/log_clustering.py
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter

from config import get_config, get_model_config

logger = logging.getLogger(__name__)


class LogClusteringModel:
    """日志聚类模型，提供多种聚类算法"""

    ALGORITHMS = {
        "kmeans": KMeans,
        "dbscan": DBSCAN,
        "hierarchical": AgglomerativeClustering
    }

    def __init__(
            self,
            algorithm: str = "kmeans",
            algorithm_params: Optional[Dict[str, Any]] = None,
            vectorizer_params: Optional[Dict[str, Any]] = None,
            random_state: int = 42,
            n_components: Optional[int] = None
    ):
        """
        初始化日志聚类模型

        Args:
            algorithm: 聚类算法名称，支持 'kmeans', 'dbscan', 'hierarchical'
            algorithm_params: 算法参数
            vectorizer_params: TF-IDF向量化器参数
            random_state: 随机种子
            n_components: 降维组件数量，如果为None则不进行降维
        """
        self.config = get_config()
        self.model_config = get_model_config()

        # 设置聚类算法
        self.algorithm_name = algorithm
        self.random_state = random_state

        # 默认算法参数
        default_params = {
            "kmeans": {
                "n_clusters": 10,
                "random_state": random_state
            },
            "dbscan": {
                "eps": 0.5,
                "min_samples": 5
            },
            "hierarchical": {
                "n_clusters": 10,
                "linkage": "ward"
            }
        }

        # 合并默认参数和用户参数
        self.algorithm_params = default_params.get(algorithm, {})
        if algorithm_params:
            self.algorithm_params.update(algorithm_params)

        # 设置向量化器参数
        default_vectorizer_params = {
            "max_features": 1000,
            "min_df": 2,
            "max_df": 0.95,
            "stop_words": "english",
            "ngram_range": (1, 2)
        }

        self.vectorizer_params = default_vectorizer_params
        if vectorizer_params:
            self.vectorizer_params.update(vectorizer_params)

        # 初始化向量化器
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)

        # 降维设置
        self.n_components = n_components
        self.dim_reducer = None

        # 初始化聚类模型
        if algorithm in self.ALGORITHMS:
            self.model = self.ALGORITHMS[algorithm](**self.algorithm_params)
        else:
            raise ValueError(f"不支持的聚类算法: {algorithm}，"
                             f"支持的算法有: {', '.join(self.ALGORITHMS.keys())}")

        # 用于存储模型状态
        self.is_fitted = False
        self.feature_matrix = None
        self.labels_ = None
        self.logs_per_cluster = None

    def fit_transform(
            self,
            texts: List[str],
            metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合模型并转换数据

        Args:
            texts: 文本列表
            metadata: 元数据列表，可用于聚类后的分析

        Returns:
            (特征矩阵, 聚类标签)
        """
        if not texts:
            logger.warning("没有文本数据用于聚类")
            return np.array([]), np.array([])

        logger.info(f"开始使用 {self.algorithm_name} 算法聚类 {len(texts)} 条日志")

        # 向量化文本
        self.feature_matrix = self.vectorizer.fit_transform(texts)

        # 降维（如果需要）
        if self.n_components and self.n_components < self.feature_matrix.shape[1]:
            if hasattr(self.feature_matrix, "toarray"):
                # 稀疏矩阵用TruncatedSVD
                self.dim_reducer = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
            else:
                # 密集矩阵用PCA
                self.dim_reducer = PCA(n_components=self.n_components, random_state=self.random_state)

            logger.info(f"将特征从 {self.feature_matrix.shape[1]} 维降至 {self.n_components} 维")
            self.feature_matrix = self.dim_reducer.fit_transform(self.feature_matrix)

        # 拟合聚类模型
        self.model.fit(self.feature_matrix)

        # 获取聚类标签
        if hasattr(self.model, "labels_"):
            self.labels_ = self.model.labels_
        else:
            logger.error(f"聚类模型 {self.algorithm_name} 没有labels_属性")
            self.labels_ = np.zeros(len(texts))

        # 分析每个聚类中的日志数量
        self.logs_per_cluster = Counter(self.labels_)

        self.is_fitted = True

        # 记录聚类结果
        n_clusters = len(set(self.labels_))
        if -1 in self.labels_:  # DBSCAN将噪声点标记为-1
            n_clusters -= 1

        logger.info(f"聚类完成，形成 {n_clusters} 个聚类")

        # 计算聚类质量
        if n_clusters > 1 and len(texts) > n_clusters:
            try:
                silhouette = silhouette_score(self.feature_matrix, self.labels_)
                logger.info(f"聚类轮廓系数: {silhouette:.4f}")
            except Exception as e:
                logger.warning(f"计算轮廓系数时出错: {str(e)}")

        return self.feature_matrix, self.labels_

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        预测新文本的聚类

        Args:
            texts: 文本列表

        Returns:
            聚类标签
        """
        if not self.is_fitted:
            logger.error("模型尚未拟合，无法进行预测")
            return np.array([])

        # 向量化新文本
        X = self.vectorizer.transform(texts)

        # 降维（如果需要）
        if self.dim_reducer:
            X = self.dim_reducer.transform(X)

        # 预测聚类
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        elif hasattr(self.model, "labels_"):
            # 某些聚类算法没有predict方法，使用最近邻居的标签
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self.feature_matrix)
            indices = nn.kneighbors(X, return_distance=False)
            return self.labels_[indices.flatten()]
        else:
            logger.error("模型不支持预测")
            return np.array([0] * len(texts))

    def get_clusters(self) -> Dict[int, List[int]]:
        """
        获取聚类结果

        Returns:
            聚类ID到样本索引的映射
        """
        if not self.is_fitted:
            return {}

        clusters = {}
        for i, label in enumerate(self.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        return clusters

    def get_important_features(self, cluster_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取特定聚类的重要特征

        Args:
            cluster_id: 聚类ID
            top_n: 返回的特征数量

        Returns:
            (特征, 重要性)元组列表
        """
        if not self.is_fitted or cluster_id not in set(self.labels_):
            return []

        # 获取该聚类的样本索引
        cluster_indices = [i for i, label in enumerate(self.labels_) if label == cluster_id]

        if not cluster_indices:
            return []

        # 获取该聚类的平均TF-IDF向量
        if hasattr(self.feature_matrix, "toarray"):
            cluster_vectors = self.feature_matrix.toarray()[cluster_indices]
        else:
            cluster_vectors = self.feature_matrix[cluster_indices]

        mean_vector = np.mean(cluster_vectors, axis=0)

        # 获取特征名称
        feature_names = self.vectorizer.get_feature_names_out()

        # 如果已降维，则无法直接获取重要特征
        if self.dim_reducer:
            return [("降维后无法识别原始特征", 0.0)]

        # 获取最重要的特征
        indices = np.argsort(mean_vector)[-top_n:]
        return [(feature_names[i], mean_vector[i]) for i in indices[::-1]]

    def visualize_clusters(
            self,
            texts: Optional[List[str]] = None,
            metadata: Optional[List[Dict[str, Any]]] = None,
            method: str = "pca"
    ) -> str:
        """
        可视化聚类结果

        Args:
            texts: 文本列表，如果已拟合则可为None
            metadata: 元数据列表，用于标记点
            method: 降维方法，'pca'或'tsne'

        Returns:
            Base64编码的PNG图像
        """
        if not self.is_fitted and texts is None:
            logger.error("未提供数据，无法可视化")
            return ""

        # 如果提供了新数据，先拟合
        if not self.is_fitted and texts is not None:
            self.fit_transform(texts, metadata)

        # 使用降维方法可视化
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=self.random_state)
        else:  # 默认使用PCA
            reducer = PCA(n_components=2, random_state=self.random_state)

        # 降维到2D
        X_2d = reducer.fit_transform(self.feature_matrix)

        # 创建图像
        plt.figure(figsize=(10, 8))

        # 获取唯一标签和颜色
        unique_labels = set(self.labels_)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        # 绘制散点图
        for label, color in zip(unique_labels, colors):
            mask = self.labels_ == label
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[color], label=f'Cluster {label}',
                alpha=0.7, s=40, edgecolors='w'
            )

        plt.legend()
        plt.title(f'日志聚类可视化 ({self.algorithm_name})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()

        # 将图保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)

        # 关闭图像，释放内存
        plt.close()

        # 返回Base64编码的图像
        return base64.b64encode(buf.read()).decode('utf-8')

    @classmethod
    def optimize_clusters(
            cls,
            texts: List[str],
            min_clusters: int = 2,
            max_clusters: int = 20,
            step: int = 1,
            algorithm: str = "kmeans",
            criterion: str = "silhouette"
    ) -> Tuple[int, float]:
        """
        优化聚类数量

        Args:
            texts: 文本列表
            min_clusters: 最小聚类数
            max_clusters: 最大聚类数
            step: 步长
            algorithm: 聚类算法
            criterion: 评价标准，支持'silhouette'或'inertia'

        Returns:
            (最佳聚类数, 最佳评分)
        """
        if not texts:
            logger.warning("没有文本数据用于优化")
            return min_clusters, 0.0

        # 向量化文本
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.95)
        X = vectorizer.fit_transform(texts)

        best_score = -np.inf if criterion == "silhouette" else np.inf
        best_n_clusters = min_clusters

        scores = []
        cluster_ranges = range(min_clusters, max_clusters + 1, step)

        logger.info(f"开始优化聚类数量，范围: {min_clusters}-{max_clusters}，算法: {algorithm}")

        for n_clusters in cluster_ranges:
            if algorithm == "kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif algorithm == "hierarchical":
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                logger.warning(f"不支持的算法: {algorithm}，使用KMeans")
                model = KMeans(n_clusters=n_clusters, random_state=42)

            labels = model.fit_predict(X)

            if criterion == "silhouette":
                try:
                    score = silhouette_score(X, labels)
                    scores.append(score)

                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except Exception as e:
                    logger.warning(f"计算轮廓系数时出错 (n_clusters={n_clusters}): {str(e)}")
                    scores.append(0)

            elif criterion == "inertia" and hasattr(model, "inertia_"):
                score = -model.inertia_  # 负的惯性，因为我们要最大化
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters

            logger.info(f"尝试 {n_clusters} 个聚类，得分: {scores[-1]:.4f}")

        logger.info(f"最佳聚类数: {best_n_clusters}，得分: {best_score:.4f}")

        # 可视化评分
        plt.figure(figsize=(10, 6))
        plt.plot(list(cluster_ranges), scores, 'o-')
        plt.xlabel('聚类数量')
        plt.ylabel('评分 (' + criterion + ')')
        plt.title('聚类数量优化')
        plt.grid(True)

        # 将图保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)

        # 关闭图像，释放内存
        plt.close()

        # 返回最佳结果
        return best_n_clusters, best_score

