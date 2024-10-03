from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

app = FastAPI()

model_name = "Eyvaz/wav2vec2-base-russian-big-kaggle"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cpu")

@app.post("/asr/")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    try:
        audio_data, _ = librosa.load(audio_file.file, sr=16000, mono=True)

        input_values = processor(audio_data, return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        return JSONResponse(content={"transcription": transcription})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=9090)