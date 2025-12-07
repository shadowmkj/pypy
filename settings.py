from pydantic_settings import BaseSettings

class AISettings(BaseSettings):
    model_name: str = "ollama:llama3.2:3b"
    embedding_model: str = "all-MiniLM-L6-v2"
AIConfig = AISettings()