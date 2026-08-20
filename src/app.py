from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

from helpers.config import get_settings
from routes.chat import router as chat_router
from routes.embed import router as embed_router
from stores.llm.LLMEnums import LLMEnums
from stores.llm.LLMProviderFactory import LLMProviderFactory
from controllers.agent import Agent
from tools import ALL_TOOLS

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    config = get_settings()
    
    factory = LLMProviderFactory(config=config)
    generation_client = factory.create(config.GENERATION_BACKEND)
    embedding_client = factory.create(config.EMBEDDING_BACKEND)

    generation_client.set_generation_model(config.GENERATION_MODEL_ID)
    embedding_client.set_embedding_model(config.EMBEDDING_MODEL_ID, config.EMBEDDING_MODEL_SIZE)


    # Attach to app.state
    app.state.llm_client = generation_client
    app.state.embedding_client = embedding_client

    app.state.agent = Agent(
        client=generation_client,
        tools=ALL_TOOLS,
        max_iterations=8
    )
    yield

app = FastAPI(
    title="Tabeeby Agent",
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(embed_router)


@app.get("/")
async def root():
    return {"message": "Server is running"}