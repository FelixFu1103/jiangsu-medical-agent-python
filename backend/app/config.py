from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    chroma_path: str = "../.chroma"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    hf_cache_dir: str = "../.models"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = False
    chunk_size: int = 500
    chunk_overlap: int = 80


@lru_cache
def get_settings() -> Settings:
    return Settings()
