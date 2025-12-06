from pydantic_settings import BaseSettings

class AISettings(BaseSettings):
    model_name: str = "ollama:ministral-3:8b"
    embedding_model: str = "all-MiniLM-L6-v2"
AIConfig = AISettings()