from typing import Dict, List, Any, Union, Tuple
import torch
import torchaudio
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoProcessor,
)
from peft import PeftModel
from pyannote.audio import Pipeline
from pyannote.core import Segment
import torch.nn.functional as F
import numpy as np
from profiles import ModelProfile
from pathlib import Path
from typing import get_args


UnpreparedAudioType = Union[BytesIO, str, Path]
PreparedAudioType = Union[np.ndarray, torch.Tensor]
AudioType = Union[UnpreparedAudioType, PreparedAudioType]


class ASRModel:
    pipeline: AutomaticSpeechRecognitionPipeline
    generate_kwargs: Dict[str, Any]
    sampling_rate: int
    use_diarization: bool
    diarization_pipeline: Pipeline

    def __init__(self, config: ModelProfile) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize ASR
        model = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_name).to(self.device)
        if config.lora_name:
            model = PeftModel.from_pretrained(model, config.lora_name)

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, **config.model_features)
        processor = AutoProcessor.from_pretrained(config.model_name, **config.model_features)
        
        forced_decoder_ids = processor.get_decoder_prompt_ids(**config.model_features)
        self.generate_kwargs = {"forced_decoder_ids": forced_decoder_ids, **config.model_features}

        self.pipeline = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=processor.feature_extractor,
            device=self.device,
        )
        self.sampling_rate = self.pipeline.feature_extractor.sampling_rate
        
        # Initialize diarization if needed
        self.use_diarization = config.use_diarization
        if self.use_diarization:
            self.diarization_pipeline = Pipeline.from_pretrained(
                config.diarization_model,
                use_auth_token=True  # You'll need Hugging Face token
            ).to(self.device)

    def preprocess(self, audio: Union[UnpreparedAudioType, List[UnpreparedAudioType]]) -> Tuple[List[np.ndarray], List[int]]:
        if not isinstance(audio, list):
            audio = [audio]

        processed = []
        sample_rates = []
        for audio_item in audio:
            audio_data, sample_rate = torchaudio.load(audio_item)
            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sampling_rate,
                )
                audio_data = resampler(audio_data)
            processed.append(audio_data.squeeze().numpy())
            sample_rates.append(sample_rate)

        return processed, sample_rates

    def transcribe(self, audio: Union[AudioType, List[AudioType]]) -> List[str]:
        if not isinstance(audio, list):
            audio = [audio]

        if any(isinstance(a, get_args(UnpreparedAudioType)) for a in audio):
            audio_data, sample_rates = self.preprocess(audio)
        else:
            audio_data = audio
            sample_rates = [self.sampling_rate] * len(audio)

        results = []
        for single_audio, sample_rate in zip(audio_data, sample_rates):
            if self.use_diarization:
                # Perform diarization
                diarization = self.diarization_pipeline({"waveform": torch.tensor(single_audio).unsqueeze(0), 
                                                       "sample_rate": sample_rate})
                
                # Split audio by speaker segments
                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_sample = int(turn.start * sample_rate)
                    end_sample = int(turn.end * sample_rate)
                    segment_audio = single_audio[start_sample:end_sample]
                    
                    # Transcribe segment
                    with torch.amp.autocast("cuda"):
                        segment_text = self.pipeline(
                            segment_audio,
                            generate_kwargs=self.generate_kwargs,
                            max_new_tokens=255
                        )["text"].strip()
                    
                    segments.append(f"[{speaker}]: {segment_text}")
                
                results.append("\n".join(segments))
            else:
                # Regular transcription without diarization
                with torch.amp.autocast("cuda"):
                    output = self.pipeline(
                        single_audio,
                        generate_kwargs=self.generate_kwargs,
                        max_new_tokens=255
                    )
                    results.append(output["text"])

        return results
