import pytest
from pathlib import Path
import numpy as np
import soundfile as sf


@pytest.fixture(scope="session", autouse=True)
def setup_test_audio():
    test_dir = Path(__file__).parent / "data"
    test_dir.mkdir(exist_ok=True)

    test_audio_path = test_dir / "test_audio.wav"
    if not test_audio_path.exists():
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        sf.write(test_audio_path, audio_data, sample_rate)

    return test_audio_path
