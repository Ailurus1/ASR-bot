from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
from io import BytesIO
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")

@app.post("/asr/")
async def transcribe_audio(audio_message: UploadFile = File(...)):
    try:
        audio_bytes = await audio_message.read()

        audio_data, sample_rate = torchaudio.load(BytesIO(audio_bytes))

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            audio_data = resampler(audio_data)

        input_features = processor(
            audio_data.squeeze().numpy(), return_tensors="pt", sampling_rate=16000
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090, log_level="debug")
