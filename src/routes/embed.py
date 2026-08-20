from typing import List, Union
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from models.enums.ResponseEnums import ResponseSignal
from routes.schemes.embed import EmbedRequest, EmbedResponse
from stores.llm.LLMInterface import LLMInterface

router = APIRouter(prefix="/embed", tags=["Embedding"])


@router.post("", response_model=EmbedResponse)
async def embed_endpoint(request: EmbedRequest, req: Request) -> EmbedResponse:

    embedding_client: LLMInterface = getattr(req.app.state, "embedding_client", None)

    if not embedding_client:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.LLM_CLIENT_NOT_INITIALIZED.value},
        )

    raw_text = request.text

    if not raw_text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.EMBEDDING_EMPTY_INPUT.value},
        )

    try:
        if isinstance(raw_text, str):
            if not raw_text.strip():
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"signal": ResponseSignal.EMBEDDING_EMPTY_INPUT.value},
                )

            vector = embedding_client.embed_text(text=raw_text)
            if vector is None:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"signal": ResponseSignal.EMBEDDING_FAILED.value},
                )

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "signal": ResponseSignal.EMBEDDING_SUCCESS.value,
                    "embeddings": vector,
                    "count": 1,
                    "dimensions": len(vector),
                },
            )

        elif isinstance(raw_text, list):
            if len(raw_text) == 0:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"signal": ResponseSignal.EMBEDDING_EMPTY_INPUT.value},
                )

            vectors = []
            for item in raw_text:
                if isinstance(item, str) and item.strip():
                    v = embedding_client.embed_text(text=item)
                    if v is None:
                        return JSONResponse(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"signal": ResponseSignal.EMBEDDING_FAILED.value},
                        )
                    vectors.append(v)

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "signal": ResponseSignal.EMBEDDING_SUCCESS.value,
                    "embeddings": vectors,
                    "count": len(vectors),
                    "dimensions": len(vectors[0]) if vectors else 0,
                },
            )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.EMBEDDING_EMPTY_INPUT.value},
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.EMBEDDING_FAILED.value},
        )
