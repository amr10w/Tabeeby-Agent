from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from models.enums.ResponseEnums import ResponseSignal
from routes.schemes.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])



@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest,req:Request) -> ChatResponse:

    client = req.app.state.llm_client

    if not client:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": ResponseSignal.LLM_CLIENT_NOT_INITIALIZED.value,
            },
        )

    ollama_res = client.generate_response(
        prompt=request.prompt,
       
    )

    if not ollama_res:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.LLM_GENERATE_ERROR.value,
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.LLM_GENERATE_SUCCESS.value,
            "response": ollama_res.message.content,
        },
    )