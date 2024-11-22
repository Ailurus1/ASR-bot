import argparse
from inference_server.profiles import PROFILES
from inference_server.model import ASRModel
from datasets import Dataset
from tqdm import tqdm
from evaluate import load
from pathlib import Path
import scipy.signal as sps
from typing import Optional
import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(".log"), logging.StreamHandler()],
)
logger = logging.getLogger()


def evaluate(dataset: Dataset, num_examples: Optional[int] = None):
    if num_examples:
        test_examples = dataset.select(range(num_examples))

    model_outputs = []
    original_transcription = []
    for id, example in tqdm(enumerate(test_examples)):
        # print(example)
        audio_data, sample_rate = (
            example["audio"]["array"],
            example["audio"]["sampling_rate"],
        )
        # audio_data, sample_rate = torchaudio.load(example["audio"])

        # if sample_rate != 16000:
        #     resampler = torchaudio.transforms.Resample(
        #         orig_freq=sample_rate, new_freq=16000
        #     )
        #     audio_data = resampler(audio_data.astype(torch.float32))

        if sample_rate != 16000:
            new_rate = 16000
            number_of_samples = round(len(audio_data) * float(new_rate) / sample_rate)
            audio_data = sps.resample(audio_data, number_of_samples)

        # transcription = transcribe(audio_data)
        # if example["transcription"] is None or transcription is None:
        #     continue
        # original_transcription.append(example["transcription"])
        # model_outputs.append(transcription)
        # if id >= num_examples:
        #     break
    wer = load("wer")

    wer_score = wer.compute(
        predictions=model_outputs, references=original_transcription
    )
    return wer_score, model_outputs, original_transcription


# wer_score, model_outputs, transcription = benchmark(dataset, 10)
# print(f'WER - {wer_score}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="classical-tiny")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".logs"))
    parser.add_argument("--input", type=Path, default=Path("datasets"))

    args = parser.parse_args()

    asr_model = ASRModel(PROFILES[args.profile])

    logger.info("Starting evaluation")
    results = evaluate("")
    logger.info("End evaluation")
    if args.save:
        logger.info("Saving results")
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"output_{formatted_time}.txt"
        with open(args.output.joinpath(filename), "w") as f:
            f.write(results)


if __name__ == "__main__":
    main()
