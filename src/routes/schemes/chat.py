from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    


class ChatResponse(BaseModel):
    signal: str
    response: Optional[str] = None