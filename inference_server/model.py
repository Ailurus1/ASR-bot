from typing import List, Union, Tuple
from io import BytesIO
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

from profiles import ModelProfile


class ASRModel:
    def __init__(self, config: ModelProfile) -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(config.model_name).to(
            self.device
        )
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
            "return_timestamps": True,
            **config.model_features,
        }

        self.pipeline = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=processor.feature_extractor,
            device=self.device,
        )
        self.sampling_rate = self.pipeline.feature_extractor.sampling_rate

        self.use_diarization = config.use_diarization
        if self.use_diarization:
            self.diarization_pipeline = Pipeline.from_pretrained(
                config.diarization_model
            ).to(self.device)

    def preprocess(
        self, audio_bytes: Union[bytes, List[bytes]]
    ) -> Tuple[torch.Tensor, int]:
        if isinstance(audio_bytes, list):
            audio_bytes = audio_bytes[0]

        audio_buffer = BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sampling_rate
            )
            waveform = resampler(waveform)

        return waveform, sample_rate

    def transcribe(self, audio: Union[bytes, List[bytes]]) -> List[str]:
        waveform, sample_rate = self.preprocess(audio)

        results = []
        if self.use_diarization:
            diarization = self.diarization_pipeline(
                {"waveform": waveform, "sample_rate": self.sampling_rate}
            )

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_sample = int(turn.start * self.sampling_rate)
                end_sample = int(turn.end * self.sampling_rate)
                segment_audio = waveform[:, start_sample:end_sample]

                with torch.amp.autocast("cuda"):
                    segment_text = self.pipeline(
                        segment_audio.squeeze().numpy(),
                        generate_kwargs=self.generate_kwargs,
                        max_new_tokens=255,
                    )["text"].strip()

                segments.append(f"[{speaker}]: {segment_text}")

            results.append("\n".join(segments))
        else:
            with torch.amp.autocast("cuda"):
                output = self.pipeline(
                    waveform.squeeze().numpy(),
                    generate_kwargs=self.generate_kwargs,
                    max_new_tokens=255,
                )

                if isinstance(output, dict) and "text" in output:
                    results.append(output["text"])
                else:
                    text = " ".join(chunk["text"] for chunk in output)
                    results.append(text)

        return results
