from .base import PreparedDataset
import polars as pl
import torchaudio
from typing import Optional


class SberDataset(PreparedDataset):
    def get_eval_dataset(
        self, sampling_rate: int, limit: Optional[int]
    ) -> pl.DataFrame:
        def return_audio_bytes(row):
            audio_data, sample_rate = torchaudio.load(row["audio"]["bytes"])
            if sample_rate != sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=sampling_rate
                )
                audio_data = resampler(audio_data)
            return audio_data.squeeze().numpy()

        if not self.dataset:
            self.load_dataset()
        test_dataset = self.dataset["test"]  # type: ignore
        if limit:
            test_dataset = test_dataset.shuffle(seed=42).select(range(limit))
        data = test_dataset.to_polars()
        audio_data = data.with_columns(
            pl.struct(pl.all()).map_elements(return_audio_bytes)
        )
        audio_data = audio_data.drop_nulls()
        return audio_data
