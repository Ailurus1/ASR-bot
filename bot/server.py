from typing import List
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler, 
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

class Bot(object):
    """
    Telegram bot which can
    read prompt and simply get
    the result from specified model
    """

    def __init__(
        self,
        token: str,
        model_endpoint: str
    ) -> None:
        self.app = ApplicationBuilder().token(token).build()
        self.model_endpoint = model_endpoint
        self.keyboard = []

            
    async def start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        reply_markup = InlineKeyboardMarkup(self.keyboard)
        await update.message.reply_text(
            "Hi! I can turn any of your audio message into text."
            "Please, send me a single voice message and I will response"
            "as soon as possible",
            reply_markup=reply_markup
        )
        
    async def help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        reply_markup = InlineKeyboardMarkup(self.keyboard)
        await update.message.reply_text(
            "Currently I can't do much, but if you send"
            "me a voice message I will trancsribe it into text",
            reply_markup=reply_markup
        )

    async def query(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Asynchronous query for getting
        text response from model service
        """
        message = await update.message.reply_text(
            f"Transcribing your audio message...",
            reply_to_message_id=update.message.message_id
        )

        text = requests.post(self.model_endpoint, files={'audio_message': message}, timeout=None)

        await context.bot.delete_message(
            chat_id=message.chat_id,
            message_id=message.message_id
        )

        await context.bot.send_message(
            chat_id=message.chat_id,
            text=text,
            read_timeout=60,
            write_timeout=60,
            pool_timeout=60
        )

    def run(self) -> None:
        """
        Running infinite polling
        """
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(
            MessageHandler(
                filters.VOICE & ~filters.COMMAND, self.query
            )
        )
        
        print("Running bot...")
        self.app.run_polling()