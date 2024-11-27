from .sber import SberDataset
from .mozila import CommonVoiceDataset

DATASETS = {
    "sber-golos-farfield": SberDataset("bond005/sberdevices_golos_100h_farfield"),
    "sber-golos-crowd": SberDataset("bond005/sberdevices_golos_10h_crowd"),
    "mozila-common-voice": CommonVoiceDataset(
        "mozilla-foundation/common_voice_17_0",
        {"name": "ru", "trust_remote_code": True},
    ),
}
