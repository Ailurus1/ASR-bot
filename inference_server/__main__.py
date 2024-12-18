from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from .model import ASRModel
from .profiles import PROFILES
import asyncio
from asyncio import Future
from asyncio.queues import Queue
from typing import Callable, List, Optional
from contextlib import asynccontextmanager
from io import BytesIO


@asynccontextmanager
async def lifespan(app: FastAPI):
    batched_server.run()
    yield


app = FastAPI(lifespan=lifespan)


class DataCollator:
    def __init__(self, stack: bool = False) -> None:
        super().__init__()
        self.stack = stack

    def collate(self, inputs: List[bytes]) -> List[bytes]:
        return inputs

    def uncollate(self, inputs: str) -> List[str]:
        return [inputs]


class BatchedServer:
    def __init__(
        self,
        inference_callable: Callable[[List[bytes]], str],
        batch_size: int,
        max_wait_time: float = 0.1,  # seconds
        collator: Optional[DataCollator] = None,
    ) -> None:
        self.queue: Queue[tuple[bytes, Future[str], float]] = Queue(
            maxsize=2 * batch_size
        )
        self.inference_callable = inference_callable
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.collator = collator if collator is not None else DataCollator()

    async def submit(self, input: bytes) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((input, future, loop.time()))
        try:
            return await asyncio.wait_for(future, timeout=20.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Request processing timed out")

    async def queue_processing(self):
        while True:
            if not self.queue.empty():
                batch = []
                futures = []

                while len(batch) < self.batch_size and not self.queue.empty():
                    try:
                        input_data, future, _ = self.queue.get_nowait()
                        batch.append(BytesIO(input_data))
                        futures.append(future)
                    except asyncio.QueueEmpty:
                        break

                if batch:
                    try:
                        results = await asyncio.to_thread(self.inference_callable, batch)
                        
                        for future, result in zip(futures, results):
                            if isinstance(result, list):
                                result = result[0]
                            future.set_result(result)
                    except Exception as e:
                        for future in futures:
                            if not future.done():
                                future.set_exception(e)

            await asyncio.sleep(0.01)

    def run(self):
        loop = asyncio.get_running_loop()
        loop.create_task(self.queue_processing())


def inference(audio_bytes):
    return app.state.asr_model.transcribe(audio_bytes)


batched_server = BatchedServer(
    inference,
    batch_size=8,
)


@app.post("/asr/")
async def transcribe_audio(audio_message: UploadFile = File(...)):
    try:
        audio_bytes = await audio_message.read()
        transcription = await batched_server.submit(audio_bytes)

        if isinstance(transcription, list):
            transcription = transcription[0] if transcription else ""

        transcription = transcription.strip()

        if not transcription:
            return JSONResponse(
                content={"error": "Could not transcribe audio (empty result)"},
                status_code=400,
            )

        response_data = {"transcription": transcription}

        return JSONResponse(
            content=response_data, media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    config = PROFILES["classical-tiny"]
    app.state.asr_model = ASRModel(config)
    uvicorn.run(app, host="0.0.0.0", port=9090, log_level="debug")
