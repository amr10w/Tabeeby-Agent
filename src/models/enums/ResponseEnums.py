from enum import Enum

class ResponseSignal(Enum):

    # LLM Signals
    LLM_GENERATE_SUCCESS = "llm_generate_success"
    LLM_CLIENT_NOT_INITIALIZED = "llm_client_not_initialized"
    LLM_GENERATE_ERROR = "llm_generate_error"
    