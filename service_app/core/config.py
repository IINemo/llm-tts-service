"""
Service configuration module.
"""

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration settings."""

    # API Settings
    api_title: str = "LLM Test-Time Scaling Service"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8080

    # CORS Settings
    allow_origins: list[str] = ["*"]
    allow_credentials: bool = True
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]

    # API Keys (loaded from environment)
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    wandb_api_key: Optional[str] = None

    # Model Settings
    default_model: str = "openai/gpt-4o-mini"
    default_strategy: str = "deepconf"
    model_cache_dir: str = os.path.expanduser("~/.cache/llm_tts_service")

    # DeepConf Defaults
    deepconf_budget: int = 8
    deepconf_window_size: int = 2048
    deepconf_filter_method: str = "top5"
    deepconf_temperature: float = 0.7

    # vLLM backend config
    vllm_model_path: Optional[str] = None  # e.g. "Qwen/Qwen2.5-Coder-7B-Instruct"
    vllm_max_model_len: int = 32000
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_seed: int = 42

    # Default TTS params
    default_scorer: str = "entropy"
    default_temperature: float = 0.6
    default_max_tokens: int = 16000
    default_thinking_mode: bool = False

    # PRM Scorer config (Qwen/Qwen2.5-Math-PRM-7B)
    prm_model_path: Optional[str] = None  # e.g. "Qwen/Qwen2.5-Math-PRM-7B"
    prm_device: str = "cuda:0"
    prm_batch_size: int = 8
    prm_torch_dtype: str = "bfloat16"
    prm_use_vllm: bool = True
    prm_gpu_memory_utilization: float = 0.3  # Lower default to fit alongside main model

    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"

    # Service Limits
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Map environment variables to snake_case
        env_prefix = ""
        case_sensitive = False
        # Also read from environment variables
        env_nested_delimiter = "__"
        # Allow extra fields from environment that aren't defined in the model
        extra = "ignore"


# Global settings instance
settings = Settings()
