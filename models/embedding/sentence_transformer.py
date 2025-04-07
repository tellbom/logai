import os
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbedding(BaseEmbeddingModel):
    """使用本地Sentence Transformer模型进行文本嵌入"""

    def __init__(
            self,
            model_name: str,
            model_path: str = "/models/sentence-transformers",
            batch_size: int = 32,
            max_seq_length: Optional[int] = None,
            device: Optional[str] = 'cpu'
    ):
        """
        初始化本地Sentence Transformer嵌入模型

        Args:
            model_name: 模型名称，如'bge-m3'
            model_path: 模型路径
            batch_size: 批处理大小
            max_seq_length: 最大序列长度
            device: 运行设备，如'cpu'或'cuda:0'，默认为自动选择
        """
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size

        # 构建完整的模型路径
        full_model_path = os.path.join(model_path, f"sentence-transformers_{model_name}")

        if not os.path.exists(full_model_path):
            logger.warning(f"模型路径 {full_model_path} 不存在，请确保您已下载模型")

        logger.info(f"正在加载本地模型: {full_model_path}")
        try:
            # 加载本地模型，禁止网络访问
            self.model = SentenceTransformer(
                model_name_or_path=full_model_path,
                device=device,
                local_files_only=True
            )

            # 设置最大序列长度
            if max_seq_length:
                self.model.max_seq_length = max_seq_length

            logger.info(f"模型 {model_name} 加载成功，向量维度: {self.get_dimension()}")

        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为嵌入向量

        Args:
            texts: 要嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        try:
            # 过滤空字符串并替换为空格（避免模型错误）
            texts = [text if text.strip() else " " for text in texts]

            # 批量生成嵌入向量
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # 转换为Python列表
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")
            # 返回零向量作为后备
            return [[0.0] * self.get_dimension()] * len(texts)

    def get_dimension(self) -> int:
        """
        获取嵌入向量的维度

        Returns:
            嵌入向量的维度
        """
        return self.model.get_sentence_embedding_dimension()