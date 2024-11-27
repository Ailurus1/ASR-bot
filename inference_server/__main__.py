from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
import torch
import logging
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import asyncio

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="russian", task="transcribe"
)

batch_size = 4

logger = logging.getLogger()
log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"

audio_queue = asyncio.Queue()
transcriptions_dict = {}

is_processing = False

async def process_audio_batch():
    global is_processing
    while True:
        audio_files_batch = []
        ids_batch = []

        for _ in range(batch_size):
            try:
                audio_bytes, request_id = await asyncio.wait_for(audio_queue.get(), timeout=10)
                audio_files_batch.append(audio_bytes)
                ids_batch.append(request_id)
            except asyncio.TimeoutError:
                break

        if audio_files_batch:
            is_processing = True
            await transcribe_audio_batch(audio_files_batch, ids_batch)
            is_processing = False

async def transcribe_audio_batch(audio_files, request_ids):
    audio_data_list = []

    for audio_bytes in audio_files:
        audio_data, sample_rate = torchaudio.load(BytesIO(audio_bytes))
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_data = resampler(audio_data)

        audio_data_list.append(audio_data.squeeze().numpy())

    input_features = processor(
        audio_data_list, return_tensors="pt", padding=True, sampling_rate=16000
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    for request_id, transcription in zip(request_ids, transcriptions):
        transcriptions_dict[request_id] = transcription

@app.post("/asr/")
async def transcribe_audio(audio_message: UploadFile = File(...)):
    audio_bytes = await audio_message.read()
    request_id = id(audio_bytes)
    await audio_queue.put((audio_bytes, request_id))

    while is_processing:
        continue

    transcription_text = transcriptions_dict.get(request_id, "Transcription is not available yet.")

    return JSONResponse(content={"transcription": transcription_text})

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(process_audio_batch())
    uvicorn.run(app, host="0.0.0.0", port=9090, log_config=log_config)
