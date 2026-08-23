from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file.

    This Pydantic settings container defines required and optional configuration
    values for the app, including API credentials, file handling limits, and
    backend selection.
    """

    APP_NAME: str
    APP_VERSION: str

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    OLLAMA_API_KEY: Optional[str] = None
    OLLAMA_API_URL: Optional[str] = None

    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None
    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = None
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = None
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = None

    VECTOR_DB_BACKEND: str
    VECTOR_DB_PATH: str
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None

    TAVILY_API_KEY: str = None
    
    class Config:
        env_file = ".env"

def get_settings()-> Settings:
    """Create and return a Settings instance.

    Returns:
        Settings: The loaded application settings object.
    """
    return Settings()