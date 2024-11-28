from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelProfile:
    model_name: str
    lora_name: Optional[str]
    hf: bool
    model_features: Dict[str, Any]


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
}
