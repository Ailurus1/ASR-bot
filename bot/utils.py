import os
import time
from telegram.ext import ContextTypes


def get_telegram_token() -> str:
    tg_token = os.getenv("TG_TOKEN", "")
    if tg_token == "":
        raise RuntimeError(
            "Error: Token for Telegram API is not specified!"
            "Please export it into environment variable `TG_TOKEN`"
        )
    return tg_token


def retry(num_retries: int, wait_time: int):
    """
    A decorator for automatic retry sending request
    """

    def _outer_wrapper(func):
        def _inner_wrapper(*args, **kwargs):
            for _ in range(num_retries):
                try:
                    result = func(*args, **kwargs)
                except Exception:
                    time.sleep(wait_time)
                else:
                    if result is not None:
                        return result

        return _inner_wrapper

    return _outer_wrapper


async def get_error_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    await context.bot.send_message(
        chat_id=chat_id,
        text=("Sorry, there was an error on our end. Please try again later."),
    )
