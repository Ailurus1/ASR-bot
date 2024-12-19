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
from .profiles import ModelProfile
from pathlib import Path
import numpy as np
from typing import get_args
from pyannote.audio import Pipeline
import torch
import logging


UnpreparedAudioType = Union[BytesIO, str, Path]
PreparedAudioType = Union[np.ndarray, torch.Tensor]
AudioType = Union[UnpreparedAudioType, PreparedAudioType]

logger = logging.getLogger(__name__)


class ASRModel:
    pipeline: AutomaticSpeechRecognitionPipeline
    generate_kwargs: Dict[str, Any]
    sampling_rate: int

    def __init__(self, config: ModelProfile) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            return_timestamps=True
        )
        self.sampling_rate = self.pipeline.feature_extractor.sampling_rate
        
        # Initialize diarization if needed
        self.use_diarization = getattr(config, 'use_diarization', False)
        if self.use_diarization:
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token="YOUR_HF_TOKEN"  # Need to be configured
            ).to(device)

    def preprocess(
        self, audio: Union[UnpreparedAudioType, List[UnpreparedAudioType]]
    ) -> Union[List[float], List[List[float]]]:
        if not isinstance(audio, list):
            audio = [audio]

        processed = []
        for audio_item in audio:
            audio_data, sample_rate = torchaudio.load(audio_item)
            
            if audio_data.shape[0] > 1:
                audio_data = audio_data.mean(dim=0, keepdim=True)
            
            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sampling_rate,
                )
                audio_data = resampler(audio_data)
            
            processed.append(audio_data.squeeze().numpy())

        return processed

    def _process_with_diarization(self, audio_path: str) -> str:
        logger.info("Starting diarization process")
        diarization = self.diarization(audio_path)
        
        # Create segments with speaker information
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        logger.info(f"Found {len(segments)} speaker segments")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])
        
        # Transcribe the full audio
        logger.info("Starting transcription")
        with torch.amp.autocast("cuda"):
            result = self.pipeline(
                audio_path,
                generate_kwargs=self.generate_kwargs,
                chunk_length_s=30,  # Add chunking for long audio
                stride_length_s=5,   # Add overlap between chunks
                return_timestamps=True
            )
        
        logger.info(f"Raw transcription result: {result}")
        
        # If we have no valid timestamps but have text, use the full text with speaker segments
        if isinstance(result, dict) and "text" in result:
            text = result["text"].strip()
            if text:
                # Split text into roughly equal parts based on speaker segments
                if segments:
                    final_text = ""
                    words = text.split()
                    words_per_segment = len(words) // len(segments)
                    
                    for i, segment in enumerate(segments):
                        start_idx = i * words_per_segment
                        end_idx = start_idx + words_per_segment if i < len(segments) - 1 else len(words)
                        segment_text = " ".join(words[start_idx:end_idx])
                        final_text += f"\n[{segment['speaker']}]: {segment_text}"
                    
                    return final_text.strip()
                else:
                    # If no speaker segments, return with UNKNOWN speaker
                    return f"[UNKNOWN]: {text}"
        
        # If we have valid chunks with timestamps, process them as before
        final_text = ""
        current_speaker = None
        
        if isinstance(result, dict):
            chunks = result.get("chunks", [])
        else:
            chunks = result
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        for chunk in chunks:
            if not isinstance(chunk, dict):
                logger.warning(f"Skipping invalid chunk: {chunk}")
                continue
            
            text = chunk.get("text", "").strip()
            if not text:
                logger.warning("Empty text in chunk")
                continue
            
            # Get timestamp safely
            timestamp = chunk.get("timestamp")
            if not timestamp or len(timestamp) != 2 or None in timestamp:
                # If no valid timestamp but we have text, try to match with the nearest speaker segment
                if segments:
                    speaker = segments[0]["speaker"]  # Use first speaker as fallback
                    if current_speaker != speaker:
                        current_speaker = speaker
                        final_text += f"\n[{current_speaker}]: "
                else:
                    if current_speaker != "UNKNOWN":
                        current_speaker = "UNKNOWN"
                        final_text += "\n[UNKNOWN]: "
            else:
                start_time, end_time = timestamp
                chunk_middle = (float(start_time) + float(end_time)) / 2
                
                # Find the speaker for this timestamp
                speaker = None
                for segment in segments:
                    if segment["start"] <= chunk_middle <= segment["end"]:
                        speaker = segment["speaker"]
                        break
                
                if speaker and speaker != current_speaker:
                    current_speaker = speaker
                    final_text += f"\n[{current_speaker}]: "
                elif not speaker and current_speaker != "UNKNOWN":
                    final_text += "\n[UNKNOWN]: "
                    current_speaker = "UNKNOWN"
            
            final_text += text + " "
        
        logger.info(f"Final text length: {len(final_text)}")
        return final_text.strip()

    def transcribe(self, audio: Union[AudioType, List[AudioType]]) -> List[str]:
        if not isinstance(audio, list):
            audio = [audio]

        if any(isinstance(a, get_args(UnpreparedAudioType)) for a in audio):
            audio = self.preprocess(audio)

        results = []
        for audio_item in audio:
            if self.use_diarization:
                # For diarization, we need to save the audio temporarily
                if isinstance(audio_item, (str, Path)):
                    audio_path = str(audio_item)
                else:
                    # Save numpy array as temporary wav file
                    temp_path = "temp_audio.wav"
                    torchaudio.save(temp_path, torch.tensor(audio_item).unsqueeze(0), self.sampling_rate)
                    audio_path = temp_path
                
                result = self._process_with_diarization(audio_path)
                results.append(result)
            else:
                with torch.amp.autocast("cuda"):
                    result = self.pipeline(
                        audio_item,
                        generate_kwargs=self.generate_kwargs,
                        max_new_tokens=255
                    )
                    results.append(result["text"])

        return results
