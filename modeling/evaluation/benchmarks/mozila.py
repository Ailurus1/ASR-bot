from .base import PreparedDataset
import polars as pl
import torchaudio
from typing import Optional


class CommonVoiceDataset(PreparedDataset):
    def get_eval_dataset(
        self, sampling_rate: int, limit: Optional[int]
    ) -> pl.DataFrame:
        def return_audio_bytes(row):
            audio_data, sample_rate = torchaudio.load(row["audio"]["path"])
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
        test_dataset = test_dataset.remove_columns(
            [
                "accent",
                "age",
                "client_id",
                "down_votes",
                "gender",
                "locale",
                "path",
                "segment",
                "up_votes",
                "variant",
            ]
        )
        data = test_dataset.to_polars()
        audio_data = data.with_columns(
            pl.struct(pl.all()).map_elements(return_audio_bytes)
        )
        audio_data = audio_data.drop_nulls()
        audio_data = audio_data.rename({"sentence": "transcription"})
        return audio_data
