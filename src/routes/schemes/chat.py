from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    chat_history:List[Dict] = None
    total_cost:int = 0
    


class ChatResponse(BaseModel):
    signal: str
    response: Optional[str] = None