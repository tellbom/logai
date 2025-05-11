# config/app_config.py
import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
import json

logger = logging.getLogger(__name__)


class AppConfig:
    """应用配置管理类，负责加载和提供配置信息"""

    def __init__(
            self,
            config_file: Optional[str] = None,
            env_prefix: str = "LOG_AI_"
    ):
        """
        初始化配置管理器

        Args:
            config_file: 配置文件路径，支持YAML或JSON格式
            env_prefix: 环境变量前缀，用于从环境变量加载配置
        """
        self.config_file = config_file
        self.env_prefix = env_prefix
        self.config = self._load_default_config()

        # 加载配置文件，如果指定了的话
        if config_file:
            file_config = self._load_from_file(config_file)
            if file_config:
                self._update_config(self.config, file_config)
                logger.info(f"已从配置文件加载配置: {config_file}")

        # 从环境变量加载配置
        env_config = self._load_from_env(env_prefix)
        if env_config:
            self._update_config(self.config, env_config)
            logger.info("已从环境变量加载配置")

    def _load_default_config(self) -> Dict[str, Any]:
        """
        加载默认配置

        Returns:
            默认配置字典
        """
        return {
            # 基础配置
            "app": {
                "name": "log-ai-analyzer",
                "log_level": "INFO",
                "output_dir": "./output",
                "temp_dir": "./temp"
            },

            #添加字段映射
            "field_mappings": {
                "timestamp_field": "@timestamp",  # 默认时间戳字段
                "message_field": "message",  # 默认消息字段
                "level_field": "log_level",  # 默认日志级别字段
                "service_field": "service_name"  # 默认服务名称字段
            },

            # elasticsearch配置 - 源库（日志数据库）
            "elasticsearch": {
                "host": "192.168.48.128",
                "port": 9200,
                "username": None,
                "password": None,
                "index_pattern": "abp-logs-*",
                "use_ssl": False,
                "verify_certs": False,
                "timeout": 60,
                "batch_size": 5000
            },

            # elasticsearch目标库配置（带IK分词器）
            "target_elasticsearch": {
                "host": "192.168.48.128",
                "port": 9200,
                "username": None,
                "password": None,
                "index": "logai-processed",
                "use_ssl": False,
                "verify_certs": False,
                "timeout": 60
            },

            # Qdrant向量数据库配置
            "qdrant": {
                "host": "192.168.48.128",
                "port": 6333,
                "grpc_port": 6334,
                "api_key": None,
                "prefer_grpc": True,
                "timeout": 60.0,
                "collection_name": "log_vectors"
            },

            # 嵌入模型配置
            "embedding": {
                "model_name": "bge-m3",
                "model_path": "/models/sentence-transformers",
                "batch_size": 32,
                "max_seq_length": 512,
                "device": "cpu"  # 'cpu' 或 'cuda:0'
            },

            # LogAI处理配置
            "logai": {
                "parsing": {
                    "algorithm": "drain",
                    "params": {
                        "sim_th": 0.4,
                        "depth": 4
                    }
                },
                "anomaly_detection": {
                    "algorithm": "isolation_forest",
                    "params": {
                        "contamination": 0.05,
                        "random_state": 42
                    }
                },
                "clustering": {
                    "algorithm": "kmeans",
                    "params": {
                        "n_clusters": 10,
                        "random_state": 42
                    }
                }
            },

            # 向量化配置
            "vectorization": {
                "chunk_size": 512,
                "batch_size": 100,
                "overlap": 0
            },

            # 检索配置
            "retrieval": {
                "hybrid_search": {
                    "es_weight": 0.3,
                    "vector_weight": 0.7,
                    "top_k": 10,
                    "rerank_url":"http://192.168.48.128:8091/rerank",  # 重排序服务URL
                }
            },
            # 日志去重配置
            "deduplication": {
                "enabled": True,
                "similarity_threshold": 0.92,  # 向量相似度阈值 (0-1)
                "novelty_threshold": 0.05,  # 语义新颖性阈值 (0-1)
                "time_window_hours": 2,  # 时间窗口（小时）
                "min_cluster_size": 2,  # 形成聚类的最小日志数
                "keep_latest_count": 3,  # 每类保留的最新日志数
                "keep_anomalies": True  # 是否保留所有异常日志
            }
        }

    def _load_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载配置

        Args:
            file_path: 配置文件路径

        Returns:
            配置字典，如果加载失败则返回None
        """
        if not os.path.exists(file_path):
            logger.warning(f"配置文件不存在: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    config = json.load(f)
                else:
                    logger.error(f"不支持的配置文件格式: {file_path}")
                    return None

                return config
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return None

    def _load_from_env(self, prefix: str) -> Dict[str, Any]:
        """
        从环境变量加载配置

        Args:
            prefix: 环境变量前缀

        Returns:
            从环境变量加载的配置
        """
        env_config = {}

        for env_name, env_value in os.environ.items():
            if env_name.startswith(prefix):
                # 移除前缀并将环境变量名转换为配置路径
                config_path = env_name[len(prefix):].lower().split('__')

                # 将环境变量值转换为适当的类型
                typed_value = self._convert_env_value(env_value)

                # 构建嵌套配置字典
                current = env_config
                for part in config_path[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                current[config_path[-1]] = typed_value

        return env_config

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool, None, List, Dict]:
        """
        将环境变量字符串值转换为适当的类型

        Args:
            value: 环境变量值

        Returns:
            转换后的值
        """
        # 检查布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # 检查None值
        if value.lower() in ('none', 'null'):
            return None

        # 检查数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # 检查JSON格式的列表或字典
        if (value.startswith('[') and value.endswith(']')) or (value.startswith('{') and value.endswith('}')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # 默认为字符串
        return value

    def _update_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        递归更新配置字典

        Args:
            target: 目标配置字典
            source: 源配置字典
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # 递归更新嵌套字典
                self._update_config(target[key], value)
            else:
                # 直接更新值
                target[key] = value

    def get(self, path: str, default: Any = None) -> Any:
        """
        获取指定路径的配置值

        Args:
            path: 配置路径，以点分隔，例如 'elasticsearch.host'
            default: 如果配置不存在，返回的默认值

        Returns:
            配置值或默认值
        """
        parts = path.split('.')
        current = self.config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, path: str, value: Any) -> None:
        """
        设置指定路径的配置值

        Args:
            path: 配置路径，以点分隔，例如 'elasticsearch.host'
            value: 要设置的值
        """
        parts = path.split('.')
        current = self.config

        # 导航到最后一级的父节点
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # 设置值
        current[parts[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置

        Returns:
            完整的配置字典
        """
        return self.config.copy()

    def save_to_file(self, file_path: str) -> bool:
        """
        将当前配置保存到文件

        Args:
            file_path: 目标文件路径

        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # 根据文件扩展名决定保存格式
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                elif file_path.endswith('.json'):
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                else:
                    logger.error(f"不支持的文件格式: {file_path}")
                    return False

            logger.info(f"配置已保存到文件: {file_path}")
            return True

        except Exception as e:
            logger.error(f"保存配置到文件时出错: {str(e)}")
            return False


# 单例模式，确保应用中只有一个配置实例
_config_instance = None


def get_config(config_file: Optional[str] = None) -> AppConfig:
    """
    获取配置实例，如果不存在则创建

    Args:
        config_file: 配置文件路径

    Returns:
        配置实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig(config_file)
    return _config_instance