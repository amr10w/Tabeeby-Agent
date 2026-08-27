import asyncio
import logging
from typing import Dict, Set
from fastapi import APIRouter, Request, Response, status
from telegram import Bot, Update

router = APIRouter(prefix="/telegram", tags=["Telegram"])
logger = logging.getLogger(__name__)


USER_SESSIONS: Dict[int, list] = {}
PROCESSED_UPDATES: Set[int] = set()


async def _keep_typing(bot: Bot, chat_id: int, stop_event: asyncio.Event):
    """Keep sending typing status every 4 seconds until the agent completes."""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


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

    # 1. Deduplication: Prevent Telegram webhook retries from running duplicate queries
    if update.update_id in PROCESSED_UPDATES:
        return Response(status_code=status.HTTP_200_OK)

    PROCESSED_UPDATES.add(update.update_id)
    if len(PROCESSED_UPDATES) > 5000:
        PROCESSED_UPDATES.clear()

    chat_id = update.message.chat_id
    user_text = update.message.text.strip()

    # 2. Handle commands
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

    # 3. Retrieve user's session history
    history = USER_SESSIONS.get(chat_id, [])

    # 4. Start continuous typing indicator in background
    stop_typing_event = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(bot, chat_id, stop_typing_event))

    # 5. Run Agent loop in a separate thread so it DOES NOT freeze the event loop
    try:
        final_answer, updated_history, cost = await asyncio.to_thread(
            agent.run,
            prompt=user_text,
            chat_history=history,
        )

        if len(updated_history) > 11:
            USER_SESSIONS[chat_id] = [updated_history[0]] + updated_history[-10:]  # 1 system prompt at index 0 + most recent 10 turns
        else:
            USER_SESSIONS[chat_id] = updated_history[-10:]

        await bot.send_message(chat_id=chat_id, text=final_answer)
    except Exception as e:
        logger.error(f"Error handling Telegram message: {e}", exc_info=True)
        await bot.send_message(
            chat_id=chat_id,
            text="I ran into an issue processing your request. Please try again."
        )
    finally:
        stop_typing_event.set()
        await typing_task

    return Response(status_code=status.HTTP_200_OK)