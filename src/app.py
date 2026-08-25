from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from telegram import Bot

from helpers.config import get_settings
from routes.chat import router as chat_router
from routes.telegram import router as telegram_router
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from agent import Agent
from tools import DoctorTools,web_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    config = get_settings()
    
    llm_factory = LLMProviderFactory(config=config)
    generation_client = llm_factory.create(config.GENERATION_BACKEND)
    embedding_client = llm_factory.create(config.EMBEDDING_BACKEND)

    generation_client.set_generation_model(config.GENERATION_MODEL_ID)
    embedding_client.set_embedding_model(
        model_id=config.EMBEDDING_MODEL_ID,
        embedding_size=config.EMBEDDING_MODEL_SIZE
    )

    vdb_factory = VectorDBProviderFactory(config=config)
    vector_db_client = vdb_factory.create(config.VECTOR_DB_BACKEND)
    if vector_db_client:
        vector_db_client.connect()

   

    app.state.llm_client = generation_client
    app.state.embedding_client = embedding_client
    app.state.vector_db_client = vector_db_client

    doctor_tool=DoctorTools(embedding_client=embedding_client,
                            vectordb_client=vector_db_client)

    ALL_TOOLS = [doctor_tool.search_doctors,web_search]

    app.state.agent = Agent(
        client=generation_client,
        tools=ALL_TOOLS,
        max_iterations=8
    )

    tg_token=config.TELEGRAM_BOT_TOKEN
    webhook_url=config.TELEGRAM_WEBHOOK_URL

    if tg_token:
        bot=Bot(token=tg_token)
        app.state.tg_bot=bot
        if webhook_url:
            await bot.set_webhook(url=webhook_url)

    yield

    # Clean shutdown

    if getattr(app.state, "tg_bot", None):
        app.state.tg_bot.delete_webhook()

    if vector_db_client:
        vector_db_client.disconnect()


app = FastAPI(
    title="Tabeeby Agent",
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(telegram_router)


@app.get("/")
async def root():
    return {"message": "Server is running"}