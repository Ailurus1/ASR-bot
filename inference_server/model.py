from typing import Dict, List, Any, Union
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
from profiles import ModelProfile
from pathlib import Path
import numpy as np
from typing import get_args


UnpreparedAudioType = Union[BytesIO, str, Path]
PreparedAudioType = Union[np.ndarray, torch.Tensor]
AudioType = Union[UnpreparedAudioType, PreparedAudioType]


class ASRModel:
    pipeline: AutomaticSpeechRecognitionPipeline
    generate_kwargs: Dict[str, Any]
    sampling_rate: int

    def __init__(self, config: ModelProfile) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not config.hf:
            assert NotImplementedError()

        model = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_name).to(device)

        if config.lora_name:
            model = PeftModel.from_pretrained(model, config.lora_name)

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, **config.model_features
        )
        processor = AutoProcessor.from_pretrained(
            config.model_name, **config.model_features
        )

        # need to check if it really improves a transcription
        forced_decoder_ids = processor.get_decoder_prompt_ids(**config.model_features)

        self.generate_kwargs = {
            "forced_decoder_ids": forced_decoder_ids,
            **config.model_features,
        }

        self.pipeline = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )
        self.sampling_rate = self.pipeline.feature_extractor.sampling_rate

    def preprocess(
        self, audio: Union[UnpreparedAudioType, List[UnpreparedAudioType]]
    ) -> Union[List[float], List[List[float]]]:
        # Handle single input
        if not isinstance(audio, list):
            audio = [audio]

        processed = []
        for audio_item in audio:
            audio_data, sample_rate = torchaudio.load(audio_item)
            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sampling_rate,
                )
                audio_data = resampler(audio_data)
            processed.append(audio_data.squeeze().numpy())

        return processed

    def transcribe(self, audio: Union[AudioType, List[AudioType]]) -> List[str]:
        # Handle single input
        if not isinstance(audio, list):
            audio = [audio]

        # Preprocess if needed
        if any(isinstance(a, get_args(UnpreparedAudioType)) for a in audio):
            audio = self.preprocess(audio)

        with torch.amp.autocast("cuda"):
            outputs = self.pipeline(
                audio, generate_kwargs=self.generate_kwargs, max_new_tokens=255
            )
            if isinstance(outputs, list):
                outputs = [output["text"] for output in outputs]
            else:
                outputs = [outputs["text"]]

        return outputs
