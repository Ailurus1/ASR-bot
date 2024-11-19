from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from model import ASRModel
import json
import pathlib


app = FastAPI()


@app.post("/asr/")
async def transcribe_audio(audio_message: UploadFile = File(...)):
    try:
        audio_bytes = await audio_message.read()
        transcription = app.state.asr_model.transcribe(audio_bytes)
        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.resolve()
    file_path = current_dir.joinpath(current_dir / "manifests" / "local.json")
    with open(file_path, "r") as f:
        config = json.load(f)
    app.state.asr_model = ASRModel(config)
    uvicorn.run(app, host="0.0.0.0", port=9090, log_level="debug")
