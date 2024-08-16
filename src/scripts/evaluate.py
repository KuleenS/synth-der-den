import argparse

import os

import random

import tomli

import numpy as np

import torch

from src.evaluate.process_i2b2 import i2b2_to_conll
from src.evaluate.process_semeval import semeval_to_conll
from src.evaluate.process_generated import generated_to_conll
from src.evaluate.process_label_generated import label_generated_conll

from src.evaluate.train_scripts import train_baseline_berts, train_baseline_generated_berts, finetune_berts

from src.evaluate.results import ood_analysis

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main(args):
    datasets = ['semeval','i2b2']

    testsets = ['semeval']

    filtering = [True]

    config_path = args.config

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    evaluate_config = config['evaluate']

    generated_note_path = evaluate_config['generated_note_path']

    results_output = evaluate_config['results_output']

    model_output = evaluate_config['model_output']

    processed_dataset_output = evaluate_config['processed_dataset_output']

    i2b2_base_path = evaluate_config['i2b2_base_path']

    semeval_path = evaluate_config['semeval_path']

    mode = evaluate_config['mode']

    semeval_to_conll(semeval_path, processed_dataset_output)
    i2b2_to_conll(i2b2_base_path, processed_dataset_output)

    # turns generated notes to conll with different filtering and modes
    for j in range(len(filtering)):
        generated_to_conll(generated_note_path, processed_dataset_output, filtering[j], mode)

    for seed in range(5):

        random_seed = random.randint(0, 100_000)

        set_seed(random_seed)

        print(f"Run {seed} {random_seed}")

        # run the baseline models
        for i in range(len(datasets)):
            print(f"training baseline {datasets[i]}")
            train_baseline_berts(processed_dataset_output, results_output, datasets[i], model_output)

        # # labels generated notes with those baseline modes
        for i in range(len(datasets)):
            print(f"labelling baseline with {datasets[i]} baseline model")
            label_generated_conll(processed_dataset_output, datasets[i], model_output)

        train_baseline_generated_berts(processed_dataset_output, results_output, model_output, "semeval")
        
        for dataset in datasets:
            finetune_berts(processed_dataset_output, results_output, model_output, dataset)
        
        ood_analysis(processed_dataset_output, results_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)