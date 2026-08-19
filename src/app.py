from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

from helpers.config import get_settings
from routes.chat import router as chat_router
from stores.llm.LLMEnums import LLMEnums
from stores.llm.LLMProviderFactory import LLMProviderFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    config = get_settings()
    
    factory = LLMProviderFactory(config=config)
    client = factory.create(LLMEnums.OLLAMA.value)
    client.set_generation_model(config.GENERATION_MODEL_ID)

    # Attach to app.state
    app.state.llm_client = client
    yield

app = FastAPI(
    title="Tabeeby Agent",
    lifespan=lifespan,
)

app.include_router(chat_router)


@app.get("/")
async def root():
    return {"message": "Server is running"}