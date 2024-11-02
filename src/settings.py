import logging
from pathlib import Path
from typing import Any, Dict

from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Directory Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed_data"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    SCRAPED_DATA_DIR: Path = DATA_DIR / "scraped_data"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Processing Settings
    MAX_TOKENS: int = 16000
    CHATBOT_MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.1
    TOP_K_RESULTS: int = 3

    # Retry Settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1
    RETRY_BACKOFF: float = 2.0

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5

    # Application Settings
    BATCH_SIZE: int = 32
    TIMEOUT: int = 30

    # Database settings
    DB_USER: str = "myapp_user"
    DB_PASSWORD: str
    DB_NAME: str = "myapp_db"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> int:
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if v.upper() not in levels:
            raise ValueError(f"Invalid log level. Must be one of {list(levels.keys())}")
        return levels[v.upper()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        for directory in [
            self.DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EMBEDDINGS_DIR,
            self.LOG_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Create settings instance
settings = Settings()
settings.create_directories()
