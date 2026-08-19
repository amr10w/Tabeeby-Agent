from enum import Enum

class LLMEnums(Enum):
    OLLAMA:str = "OLLAMA"

class OllamaEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"