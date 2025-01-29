import argparse

import os

from pathlib import Path

import shutil

def main(args):

    input_folder = args.input
    output_folder = args.output

    list_of_pipe_files = [input_folder / x for x in os.listdir(input_folder) if x.find('pipe') >= 0]
    
    os.makedirs(output_folder / 'train', exist_ok=True)
    os.makedirs(output_folder / 'dev', exist_ok=True)
    os.makedirs(output_folder / 'test', exist_ok=True)

    split_1 = int(0.8 * len(list_of_pipe_files))
    split_2 = int(0.9 * len(list_of_pipe_files))

    train_filenames = list_of_pipe_files[:split_1]
    dev_filenames = list_of_pipe_files[split_1:split_2]
    test_filenames = list_of_pipe_files[split_2:]

    for split_files, split_name in zip([train_filenames, dev_filenames, test_filenames] ,["train", "dev", "test"]):
        for file in split_files:
            shutil.copy(file, output_folder /  split_name / file.name)
            shutil.copy(str(file).replace(".pipe", ".text"), output_folder /  split_name / file.name.replace(".pipe", ".text"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    main(args)