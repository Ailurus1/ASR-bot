import argparse
from inference_server.profiles import PROFILES
from inference_server.model import ASRModel
from tqdm import tqdm
from evaluate import load
from pathlib import Path
from typing import Optional
import datetime
import logging
import polars as pl
from typing import Tuple, List
import json
from modeling.evaluation.benchmarks import DATASETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(".log"), logging.StreamHandler()],
)
logger = logging.getLogger()


def _split_into_batches(df: pl.DataFrame, batch_size: int) -> List[pl.DataFrame]:
    num_rows = df.shape[0]
    num_batches = (num_rows + batch_size - 1) // batch_size
    batches = [df[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]
    return batches


def evaluate(
    model: ASRModel,
    dataset: pl.DataFrame,
    batch_size: int = 4,
    num_examples: Optional[int] = 100,
) -> Tuple[float, List[str], List[str]]:
    if num_examples:
        dataset = dataset.sample(num_examples, seed=42)
    batches = _split_into_batches(dataset, batch_size)

    model_outputs = []
    original_transcription = []
    for batch in tqdm(batches):
        transcriptions = model.transcribe(batch["audio"].to_list())
        original_transcription.extend(batch["transcription"].to_list())
        model_outputs.extend(transcriptions)

    wer = load("wer")
    wer_score = wer.compute(
        predictions=model_outputs, references=original_transcription
    )
    return wer_score, model_outputs, original_transcription


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_profile", type=str, default="classical-tiny")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".logs"))
    parser.add_argument("--dataset_profile", type=str, default="sber-golos-farfield")

    args = parser.parse_args()

    asr_model = ASRModel(PROFILES[args.model_profile])
    dataset = DATASETS[args.dataset_profile]

    logger.info("Get evaluation dataset")
    audio_data = dataset.get_eval_dataset(asr_model.sampling_rate)

    logger.info("Start evaluation")
    wer_score, model_outputs, transcriptions = evaluate(
        asr_model, audio_data, args.batch, args.limit
    )
    logger.info("End evaluation")

    if args.save:
        args.output.mkdir(parents=True, exist_ok=True)
        logger.info("Saving results")
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        result_filename = f"result_{formatted_time}.json"
        with open(args.output.joinpath(result_filename), "w") as f:
            json.dump(
                {
                    "model": args.model_profile,
                    "wer": wer_score,
                    "dataset": args.dataset_profile,
                },
                f,
            )
        artifacts = f"artifacts_{formatted_time}.json"
        with open(args.output.joinpath(artifacts), "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"model_output": output, "original_transcription": transcription}
                    for output, transcription in zip(model_outputs, transcriptions)
                ],
                f,
                ensure_ascii=False,
                indent=4,
            )
    print(
        "model:",
        args.model_profile,
        "wer:",
        wer_score,
        "dataset:",
        args.dataset_profile,
    )


if __name__ == "__main__":
    main()
