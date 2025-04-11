# config/__init__.py
from .app_config import AppConfig, get_config
from .model_config import ModelConfig, get_model_config

__all__ = ['AppConfig', 'get_config', 'ModelConfig', 'get_model_config']