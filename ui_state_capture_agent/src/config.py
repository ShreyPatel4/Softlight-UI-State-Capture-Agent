from __future__ import annotations

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/uistate"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str = "ui-state-capture"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    headless: bool = True


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
