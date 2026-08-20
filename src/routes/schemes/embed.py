from typing import List, Optional, Union
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    text: Union[str, List[str]]


class EmbedResponse(BaseModel):
    signal: str
    embeddings: Optional[Union[List[float], List[List[float]]]] = None
    count: Optional[int] = None
    dimensions: Optional[int] = None
