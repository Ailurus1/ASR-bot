import argparse
import torch
import torchaudio
import torchaudio.functional as F
import pandas as pd
import os

def cli():
    parser = argparse.ArgumentParser(description='Arguments with dataset_name, augment_type and augment_params')
    parser.add_argument(
        '--dataset_name',
        '-df',
        type=str,
        required=True,
        help='Name of dataset')
    parser.add_argument(
        '--augment_type',
        '-aug_type',
        type=str,
        required=True,
        help='Type of augmentation')
    parser.add_argument(
        '--augment_params',
        '-aug_param',
        type=float,
        required=False,
        help='Params for augmentation')
    parser.add_argument(
        '--input_path',
        '-inp',
        type=str,
        required=True,
        help='path/to/orig_audio')
    return parser.parse_args()

def add_pause(filename, data_path):
    audio, sample_rate = torchaudio.load(rf'{data_path}\{filename}')
    pause_time = 5
    changed_audio = torch.cat((audio[0][:len(audio[0])//2],torch.zeros(sample_rate*pause_time),audio[0][len(audio[0])//2:])).unsqueeze(0)
    out_path = '/'.join(str(el) for el in data_path.split('/')[:-1])
    last_folder = data_path.split('/')[-1]
    torchaudio.save(f"{out_path}/pause_{last_folder}/pause_{filename}", changed_audio, sample_rate, format="wav")

def speed_change_audio(filename, speedup, data_path):
    audio, sample_rate = torchaudio.load(rf'{data_path}\{filename}')
    changed_audio, _= torchaudio.functional.speed(audio, sample_rate, speedup)
    out_path = '/'.join(str(el) for el in data_path.split('/')[:-1])
    last_folder = data_path.split('/')[-1]
    torchaudio.save(f"{out_path}/speedup_{last_folder}/speedup_{speedup}_{filename}", changed_audio, sample_rate, format="wav")

def main(args):
    type= args.augment_type
    input_path = args.input_path

    if type == 'speedchange':
        speedup = args.augment_params

        last_folder = input_path.split('/')[-1]
        audio_files = pd.Series(os.listdir(input_path))
        out_path = '/'.join(str(el) for el in input_path.split('/')[:-1])
        new_folder = f'{out_path}/speedup_{last_folder}'
        if not os.path.exists(new_folder):
            os.mkdir(f'{out_path}/speedup_{last_folder}')
        audio_files.apply(lambda row: speed_change_audio(row, speedup, input_path))

    if type == 'add_pause':
        last_folder = input_path.split('/')[-1]
        audio_files = pd.Series(os.listdir(input_path))
        out_path = '/'.join(str(el) for el in input_path.split('/')[:-1])
        new_folder = f'{out_path}/pause_{last_folder}'
        if not os.path.exists(new_folder):
            os.mkdir(f'{out_path}/pause_{last_folder}')
        audio_files.apply(lambda row: add_pause(row, input_path))


if __name__ == '__main__':
    dict_args = cli()
    main(dict_args)