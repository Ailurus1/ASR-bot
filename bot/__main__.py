import os

from server import Bot
from utils import get_telegram_token


def launch_bot(telegram_token: str) -> None:
    model_endpoint_host = "inference-service" if os.environ.get("CONTAINERIZED", "") == "true" else "0.0.0.0"
    app = Bot(
        telegram_token,
        model_endpoint=f"http://{model_endpoint_host}:9090/asr/"
    )
    app.run()


def main() -> None:
    launch_bot(
        telegram_token=get_telegram_token(),
    )


if __name__ == "__main__":
    main()
