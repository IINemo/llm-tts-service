"""Telegram bot that replies with the user's chat ID on /start.

Run once to let team members discover their chat ID:
    python -m llm_tts.utils.telegram_bot

Environment variables (or .env file):
    TELEGRAM_BOT_TOKEN — Telegram Bot API token (required)
"""

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"Your chat ID:\n\n<code>{chat_id}</code>\n\n"
        "Set this in your .env as TELEGRAM_CHAT_ID.",
        parse_mode="HTML",
    )
    log.info(f"/start from chat_id={chat_id}")


def main() -> None:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required")

    application = (
        ApplicationBuilder()
        .token(token)
        .connect_timeout(30)
        .read_timeout(30)
        .build()
    )
    application.add_handler(CommandHandler("start", start_handler))
    log.info("Bot started — send /start to get your chat ID")
    application.run_polling()


if __name__ == "__main__":
    main()
