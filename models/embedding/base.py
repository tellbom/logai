from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any


class BaseEmbeddingModel(ABC):
    """嵌入模型的基类，定义了所有嵌入模型必须实现的接口"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为嵌入向量列表

        Args:
            texts: 要嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        获取嵌入向量的维度

        Returns:
            嵌入向量的维度
        """
        pass
