from abc import ABC,abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

class LLMInterface(ABC):
    @abstractmethod
    def set_generation_model(self,model_id:str):
        pass

    @abstractmethod
    def generate_response(
        self, 
        prompt: str,
        chat_history: Optional[List[Dict[str,Any]]]= None, 
        max_output_tokens: Optional[int]=None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        )->Any:
        pass


    @abstractmethod
    def construct_prompt(self, prompt: str, role: str,truncate: bool = False):
        pass