from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelProfile:
    model_name: str
    lora_name: Optional[str]
    hf: bool
    model_features: Dict[str, Any]
    use_diarization: bool = False
    diarization_model: str = "pyannote/speaker-diarization-3.1"


PROFILES = {
    "classical-tiny": ModelProfile(
        model_name="openai/whisper-tiny",
        lora_name=None,
        hf=True,
        model_features={"language": "russian", "task": "transcribe"},
    ),
    "tiny-lora-farfield": ModelProfile(
        model_name="openai/whisper-tiny",
        lora_name="RedCaesar/lora-adapter-sber-farfield",
        hf=True,
        model_features={"language": "russian", "task": "transcribe"},
    ),
    "tiny-lora-crowd": ModelProfile(
        model_name="openai/whisper-tiny",
        lora_name="RedCaesar/lora-adapter-sber-crowd",
        hf=True,
        model_features={"language": "russian", "task": "transcribe"},
    ),
    "tiny-lora-single": ModelProfile(
        model_name="openai/whisper-tiny",
        lora_name="RedCaesar/lora-adapter-single-voice",
        hf=True,
        model_features={"language": "russian", "task": "transcribe"},
    ),
    "tiny-with-diarization": ModelProfile(
        model_name="openai/whisper-tiny",
        lora_name=None,
        hf=True,
        model_features={"language": "russian", "task": "transcribe"},
        use_diarization=True
    ),
    "tiny-finetuned-ru-golos-sova-common_voice": ModelProfile(
        model_name="Ailurus/whisper-tiny-finetuned-ru",
        lora_name=None,
        hf=True,
        model_features={"language": "russian", "task": "transcribe"},
    ),
    "tiny-finetuned-ru-golos-sova-common_voice-with-diarization": ModelProfile(
        model_name="Ailurus/whisper-tiny-finetuned-ru",
        lora_name=None,
        hf=True,
        model_features={"language": "russian", "task": "transcribe"},
        use_diarization=True
    ),
}
