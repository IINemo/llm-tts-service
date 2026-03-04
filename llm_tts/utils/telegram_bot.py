"""Telegram bot that replies with the user's chat ID on /start.

Run once to let team members discover their chat ID:
    python -m llm_tts.utils.telegram_bot

Environment variables (or .env file):
    TELEGRAM_BOT_TOKEN — Telegram Bot API token (required)
    TELEGRAM_BOT_LOG_DIR — log directory (default: ./telegram_bot_logs)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _setup_logging() -> None:
    log_dir = Path(os.environ.get("TELEGRAM_BOT_LOG_DIR", "./telegram_bot_logs"))
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = log_dir / today
    day_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # output.log — INFO+
    out_handler = logging.FileHandler(day_dir / "output.log", encoding="utf-8")
    out_handler.setLevel(logging.INFO)
    out_handler.setFormatter(formatter)
    root.addHandler(out_handler)

    # stderr.log — WARNING+
    err_handler = logging.FileHandler(day_dir / "stderr.log", encoding="utf-8")
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    root.addHandler(err_handler)


log = logging.getLogger(__name__)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user = update.effective_user
    username = f"@{user.username}" if user.username else user.full_name
    await update.message.reply_text(
        f"Your chat ID:\n\n<code>{chat_id}</code>\n\n"
        "Set this in your .env as TELEGRAM_CHAT_ID.",
        parse_mode="HTML",
    )
    log.info(f"/start from {username} (chat_id={chat_id})")


def main() -> None:
    load_dotenv()
    _setup_logging()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required")

    application = (
        ApplicationBuilder().token(token).connect_timeout(30).read_timeout(30).build()
    )
    application.add_handler(CommandHandler("start", start_handler))
    log.info("Bot started — send /start to get your chat ID")
    application.run_polling()


if __name__ == "__main__":
    main()
