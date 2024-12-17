from io import BytesIO
from typing import List, Any
import json

import requests
from telegram import Update, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from moviepy.editor import VideoFileClip
from utils import get_error_message


class Bot(object):
    def __init__(
        self, token: str, model_endpoint: str = "http://localhost:9090/asr/"
    ) -> None:
        self.app = ApplicationBuilder().token(token).build()
        self.model_endpoint = model_endpoint
        self.keyboard: List[Any] = []

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Check if it's a private chat or group
        if update.effective_chat.type == "private":
            reply_markup = InlineKeyboardMarkup(self.keyboard)
            await update.message.reply_text(
                "Hi! I can turn any of your audio or video messages into text. "
                "Please, send me a single voice message and I will respond "
                "as soon as possible",
                reply_markup=reply_markup,
            )
        else:
            await update.message.reply_text(
                "Hi! I'm ready to transcribe voice messages in this group. "
                "Just send a voice message and I'll transcribe it!"
            )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_chat.type == "private":
            reply_markup = InlineKeyboardMarkup(self.keyboard)
            await update.message.reply_text(
                "Currently I can't do much, but if you send "
                "me a voice message I will transcribe it into text",
                reply_markup=reply_markup,
            )
        else:
            await update.message.reply_text(
                "I can transcribe voice messages in this group. "
                "Just send a voice message!"
            )

    async def query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Asynchronous query for getting
        text response from model service
        triggered by any audio message
        """
        message = await update.message.reply_text(
            "Transcribing your audio message...",
            reply_to_message_id=update.message.message_id,
        )
        if update.message.voice is not None:
            audio = await update.message.voice.get_file()
            audio_bytes = BytesIO(await audio.download_as_bytearray())
        elif update.message.video_note is not None:
            video_note = await update.message.video_note.get_file()
            byte_data = await video_note.download_as_bytearray()
            with open("video_note.mp4", "wb") as video_file:
                video_file.write(byte_data)

            audio_clip = VideoFileClip("video_note.mp4").audio
            audio_clip.write_audiofile("audio.oga", codec="libvorbis")

            with open("audio.oga", "rb") as file:
                byte_data = file.read()
            audio_bytes = BytesIO(byte_data)

        try:
            raw_response = requests.post(
                self.model_endpoint,
                files={"audio_message": ("audio_message.wav", audio_bytes)},
                timeout=None,
            )
        except Exception as exc:
            await get_error_message(context, message.chat_id)
            print(exc)
            raise RuntimeError(f"Error with the ASR service. Got {exc}")

        try:
            response_data = json.loads(raw_response.text)
            if "error" in response_data:
                raise RuntimeError(f"ASR service error: {response_data['error']}")

            text = response_data["transcription"]
            if not isinstance(text, str):
                raise RuntimeError(f"Unexpected transcription format: {text}")

        except json.JSONDecodeError as exc:
            await get_error_message(context, message.chat_id)
            print(f"JSON decode error: {exc}")
            print(f"Raw response: {raw_response.text}")
            raise RuntimeError("Invalid JSON response from ASR service")
        except Exception as exc:
            await get_error_message(context, message.chat_id)
            print(exc)
            raise RuntimeError(
                f"Expected to get `transcription` field in "
                f"response. Got {raw_response.text}"
            )

        await context.bot.delete_message(
            chat_id=message.chat_id, message_id=message.message_id
        )

        await context.bot.send_message(
            chat_id=message.chat_id,
            text=text,
            reply_to_message_id=update.message.message_id,
        )

    def run(self) -> None:
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(
            MessageHandler(
                filters.VOICE & ~filters.COMMAND,
                self.query,
            )
        )
        self.app.add_handler(
            MessageHandler(
                filters.VIDEO_NOTE & ~filters.COMMAND,
                self.query
            )
        )

        print("Running bot...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)
