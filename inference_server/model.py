from dataclasses import dataclass
from typing import Dict, List, Any
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoProcessor,
)
from peft import PeftModel
from io import BytesIO
import torchaudio


@dataclass
class ASRConfig:
    model_name: str
    lora: str
    hf: bool
    model_features: Dict[str, str]


class ASRModel:
    pipeline: AutomaticSpeechRecognitionPipeline
    generate_kwargs: Dict[str, Any]

    def __init__(self, config: Dict[str, Any]) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not config["hf"]:
            assert "Not implemented"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(config["model_name"]).to(
            device
        )

        if config["lora"]:
            model = PeftModel.from_pretrained(model, config["lora"])

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], **config["model_features"]
        )
        processor = AutoProcessor.from_pretrained(
            config["model_name"], **config["model_features"]
        )

        # need to check if it really improves a transcription
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            **config["model_features"]
        )

        self.generate_kwargs = {
            "forced_decoder_ids": forced_decoder_ids,
            **config["model_features"],
        }

        self.pipeline = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )

    def preprocess(self, audio_bytes) -> List[float]:
        audio_data, sample_rate = torchaudio.load(audio_bytes)

        if sample_rate != self.pipeline.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.pipeline.feature_extractor.sampling_rate,
            )
            audio_data = resampler(audio_data)
        return audio_data.squeeze().numpy()

    def transcribe(self, audio_bytes: BytesIO) -> List[str]:
        audio = self.preprocess(audio_bytes)
        with torch.cuda.amp.autocast():
            text = self.pipeline(
                audio, generate_kwargs=self.generate_kwargs, max_new_tokens=255
            )["text"]
        return [text]
