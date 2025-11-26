from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/uistate"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str = "ui-state-capture"
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    hf_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    transformers_cache: str | None = None
    hf_home: str | None = None
    headless: bool = True
    max_steps: int = 10
    dom_diff_threshold: float = 0.05
    max_action_failures: int = 2

    
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
