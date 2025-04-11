# config/model_config.py
import os
import logging
from typing import Dict, Any, Optional, List, Union
import json

from .app_config import get_config

logger = logging.getLogger(__name__)


class ModelConfig:
    """
    模型配置管理类，专门管理与AI模型相关的配置
    """

    def __init__(self, app_config=None):
        """
        初始化模型配置

        Args:
            app_config: 应用配置实例，如果为None则自动获取
        """
        self.app_config = app_config or get_config()
        self.embedding_config = self._get_embedding_config()
        self.logai_config = self._get_logai_config()
        self.vectorization_config = self._get_vectorization_config()

    def _get_embedding_config(self) -> Dict[str, Any]:
        """
        获取嵌入模型配置

        Returns:
            嵌入模型配置字典
        """
        embedding_config = self.app_config.get("embedding", {})

        # 设置默认值
        default_config = {
            "model_name": "bge-m3",
            "model_path": "/models/sentence-transformers",
            "batch_size": 32,
            "max_seq_length": 512,
            "device": "cpu"
        }

        # 合并配置
        for key, default_value in default_config.items():
            if key not in embedding_config:
                embedding_config[key] = default_value

        # 确保模型路径存在
        model_path = embedding_config["model_path"]
        model_name = embedding_config["model_name"]
        full_model_path = os.path.join(model_path, f"sentence-transformers_{model_name}")

        if not os.path.exists(full_model_path):
            logger.warning(f"模型路径不存在: {full_model_path}")
            logger.warning("请确保已下载模型或检查配置")

        return embedding_config

    def _get_logai_config(self) -> Dict[str, Any]:
        """
        获取LogAI处理配置

        Returns:
            LogAI配置字典
        """
        logai_config = self.app_config.get("logai", {})

        # 设置默认的解析配置
        if "parsing" not in logai_config:
            logai_config["parsing"] = {
                "algorithm": "drain",
                "params": {
                    "sim_th": 0.4,
                    "depth": 4
                }
            }

        # 设置默认的异常检测配置
        if "anomaly_detection" not in logai_config:
            logai_config["anomaly_detection"] = {
                "algorithm": "isolation_forest",
                "params": {
                    "contamination": 0.05,
                    "random_state": 42
                }
            }

        # 设置默认的聚类配置
        if "clustering" not in logai_config:
            logai_config["clustering"] = {
                "algorithm": "kmeans",
                "params": {
                    "n_clusters": 10,
                    "random_state": 42
                }
            }

        return logai_config

    def _get_vectorization_config(self) -> Dict[str, Any]:
        """
        获取向量化配置

        Returns:
            向量化配置字典
        """
        vectorization_config = self.app_config.get("vectorization", {})

        # 设置默认值
        default_config = {
            "chunk_size": 512,
            "batch_size": 100,
            "overlap": 0
        }

        # 合并配置
        for key, default_value in default_config.items():
            if key not in vectorization_config:
                vectorization_config[key] = default_value

        return vectorization_config

    def get_embedding_model_path(self) -> str:
        """
        获取完整的嵌入模型路径

        Returns:
            模型路径
        """
        model_path = self.embedding_config["model_path"]
        model_name = self.embedding_config["model_name"]
        return os.path.join(model_path, f"sentence-transformers_{model_name}")

    def get_device(self) -> str:
        """
        获取模型运行设备

        Returns:
            设备名称，如'cpu'或'cuda:0'
        """
        device = self.embedding_config.get("device", "cpu")

        # 如果指定了使用CUDA，但系统没有CUDA，则回退到CPU
        if device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(f"CUDA不可用，回退到CPU")
                    return "cpu"
            except ImportError:
                logger.warning(f"未安装PyTorch或无法导入，回退到CPU")
                return "cpu"

        return device

    def get_parse_algorithm(self) -> Dict[str, Any]:
        """
        获取日志解析算法配置

        Returns:
            包含算法名称和参数的字典
        """
        parse_config = self.logai_config.get("parsing", {})
        return {
            "algorithm": parse_config.get("algorithm", "drain"),
            "params": parse_config.get("params", {"sim_th": 0.4, "depth": 4})
        }

    def get_anomaly_detection_algorithm(self) -> Dict[str, Any]:
        """
        获取异常检测算法配置

        Returns:
            包含算法名称和参数的字典
        """
        anomaly_config = self.logai_config.get("anomaly_detection", {})
        return {
            "algorithm": anomaly_config.get("algorithm", "isolation_forest"),
            "params": anomaly_config.get("params", {"contamination": 0.05, "random_state": 42})
        }

    def get_clustering_algorithm(self) -> Dict[str, Any]:
        """
        获取聚类算法配置

        Returns:
            包含算法名称和参数的字典
        """
        clustering_config = self.logai_config.get("clustering", {})
        return {
            "algorithm": clustering_config.get("algorithm", "kmeans"),
            "params": clustering_config.get("params", {"n_clusters": 10, "random_state": 42})
        }

    def update_model_parameters(self, model_type: str, params: Dict[str, Any]) -> None:
        """
        更新模型参数

        Args:
            model_type: 模型类型，可以是'embedding', 'parsing', 'anomaly_detection', 'clustering'
            params: 要更新的参数字典
        """
        if model_type == "embedding":
            self.embedding_config.update(params)
        elif model_type in ["parsing", "anomaly_detection", "clustering"]:
            if model_type in self.logai_config:
                if "params" in params:
                    # 更新参数
                    self.logai_config[model_type]["params"].update(params["params"])

                # 更新其他配置
                for key, value in params.items():
                    if key != "params":
                        self.logai_config[model_type][key] = value
            else:
                self.logai_config[model_type] = params
        else:
            logger.warning(f"未知的模型类型: {model_type}")


# 单例模式实现
_model_config_instance = None


def get_model_config() -> ModelConfig:
    """
    获取模型配置实例

    Returns:
        模型配置实例
    """
    global _model_config_instance
    if _model_config_instance is None:
        _model_config_instance = ModelConfig()
    return _model_config_instance