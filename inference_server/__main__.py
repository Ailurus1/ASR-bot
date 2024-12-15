from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import torch
from model import ASRModel
from profiles import PROFILES
import asyncio
from asyncio import Future
from asyncio.queues import Queue
from typing import Callable, Generic, TypeVar, Tuple, List, Optional

app = FastAPI()

T = TypeVar("T")
U = TypeVar("U")

class DataCollator:
    def __init__(self, stack: bool = False) -> None:
        super().__init__()
        self.stack = stack

    def collate(self, inputs: List[T]) -> T:
        if self.stack is not None:
            return torch.stack(inputs)

        return torch.cat(inputs)

    def uncollate(self, input: T) -> List[T]:
        return [x if self.stack else x.unsqueeze(0) for x in input]


class BatchedServer(Generic[T, U]):
    def __init__(
        self,
        inference_callable: Callable[[T], U],
        batch_size: int,
        collator: Optional[DataCollator[T]] = None,
    ) -> None:
        self.queue: Queue[Tuple[T, Future[U], float]] = Queue(
            maxsize=2 * batch_size
        )
        self.inference_callable = inference_callable
        self.batch_size = batch_size
        self.collator = collator if collator is not None else DataCollator()

    async def submit(self, input: T) -> U:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((input, future, loop.time()))
        return await future

    async def queue_processing(self):
        loop = asyncio.get_running_loop()

        while True:
            if not self.queue.empty():
                batch_size = min(self.queue.qsize(), self.batch_size)
                inputs_list: List[T] = []
                futures: List[Future[U]] = []
                for _ in range(batch_size):
                    input, future, _ = self.queue.get_nowait()
                    inputs_list.append(input)
                    futures.append(future)
                inputs = self.collator.collate(inputs_list)

                try:
                    outputs = await asyncio.to_thread(self.inference_callable, inputs)
                    outputs_list = self.collator.uncollate(outputs)
                    for output, future in zip(outputs_list, futures):
                        future.set_result(output)
                except Exception as exc:
                    for future in futures:
                        future.set_exception(Exception("Could not process batch"))
            else:
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
        transcription = batched_server.submit(audio_bytes)
        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    config = PROFILES["classical-tiny"]
    app.state.asr_model = ASRModel(config)
    uvicorn.run(app, host="0.0.0.0", port=9090, log_level="debug")
