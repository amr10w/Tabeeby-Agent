import asyncio
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from models.enums.ResponseEnums import ResponseSignal
from routes.schemes.chat import ChatRequest, ChatResponse
from agent import Agent
router = APIRouter(prefix="/chat", tags=["Chat"])



@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request) -> ChatResponse:

    agent: Agent = getattr(req.app.state, "agent", None)

    if not agent:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.LLM_CLIENT_NOT_INITIALIZED.value},
        )

    # The agent runs the loop in a worker thread to prevent event loop blocking
    final_text, history, cost = await asyncio.to_thread(
        agent.run,
        prompt=request.prompt,
        chat_history=request.chat_history,
        total_cost=request.total_cost,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.LLM_GENERATE_SUCCESS.value,
            "response": final_text,
        },
    )