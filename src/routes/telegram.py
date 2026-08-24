import logging
from typing import Any, Dict
from fastapi import APIRouter, Request, Response, status
from telegram import Bot, Update

router = APIRouter(prefix="/telegram", tags=["Telegram"])
logger = logging.getLogger(__name__)

# Simple in-memory session history mapped by Telegram chat_id
# (Replace with database/Redis for multi-worker production)
USER_SESSIONS: Dict[int, list] = {}


@router.post("/webhook")
async def telegram_webhook(request: Request):
    bot: Bot = getattr(request.app.state, "tg_bot", None)
    agent = getattr(request.app.state, "agent", None)

    if not bot or not agent:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    data = await request.json()
    update = Update.de_json(data, bot)

    if not update or not update.message or not update.message.text:
        return Response(status_code=status.HTTP_200_OK)

    chat_id = update.message.chat_id
    user_text = update.message.text

    # 1. Handle commands
    if user_text.startswith("/start"):
        USER_SESSIONS[chat_id] = []
        await bot.send_message(
            chat_id=chat_id,
            text="Hello! I am Tabeeby, your medical assistant. How can I help you find a doctor today?"
        )
        return Response(status_code=status.HTTP_200_OK)

    if user_text.startswith("/reset"):
        USER_SESSIONS[chat_id] = []
        await bot.send_message(chat_id=chat_id, text="Conversation history reset.")
        return Response(status_code=status.HTTP_200_OK)

    # 2. Retrieve user's session history
    history = USER_SESSIONS.get(chat_id, [])

    # 3. Notify user agent is thinking
    await bot.send_chat_action(chat_id=chat_id, action="typing")

    # 4. Run Agent loop
    try:
        final_answer, updated_history, cost = agent.run(
            prompt=user_text,
            chat_history=history,
        )
        USER_SESSIONS[chat_id] = updated_history[-10:]  # Keep last 10 turns
        
        await bot.send_message(chat_id=chat_id, text=final_answer)
    except Exception as e:
        logger.error(f"Error handling Telegram message: {e}")
        await bot.send_message(
            chat_id=chat_id,
            text="I ran into an issue processing your request. Please try again."
        )

    return Response(status_code=status.HTTP_200_OK)