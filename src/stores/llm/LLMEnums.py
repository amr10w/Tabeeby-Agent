from enum import Enum

class LLMEnums(Enum):
    OLLAMA = "OLLAMA"
    OLLAMAE = "OLLAMAE"

class OllamaEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"