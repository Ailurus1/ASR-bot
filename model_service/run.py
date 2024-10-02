import argparse

import uvicorn
from fastapi import FastAPI

from api import asr


def parse_args():
    parser = argparse.ArgumentParser(description="Launch service")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    return args


def run_server():
    args = parse_args()

    app = FastAPI()
    app.include_router(asr.router)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    run_server()
