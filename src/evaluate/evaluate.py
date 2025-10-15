import argparse

import csv

import os

import pathlib

import random

import tomli

import polars as pl

import numpy as np

import torch

import mysql.connector as mariadb

from src.evaluate.process_ncbi import ncbi_to_conll
from src.evaluate.process_bc5dr import bc5dr_to_conll
from src.evaluate.process_semeval import semeval_to_conll
from src.evaluate.process_generated import generated_to_conll
from src.evaluate.process_label_generated import label_generated_conll, combine_with_generated

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
    datasets = ['semeval','bc5dr', 'ncbi']

    config_path = args.config

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    evaluate_config = config['evaluate']

    generated_note_paths = evaluate_config['generated_note_paths']

    results_output = evaluate_config['results_output']

    model_output = evaluate_config['model_output']

    processed_dataset_output = evaluate_config['processed_dataset_output']

    semeval_path = evaluate_config['semeval_path']

    mode = evaluate_config['mode']

    mixed_training = evaluate_config["mixed_training"]

    mrconso =  pathlib.Path("/home/ksasse/umls/2024AB/META") / "MRCONSO.RRF"

    MRCONSO = pl.read_csv(mrconso, separator="|", has_header=False, encoding="utf8", quote_char=None)
    MRCONSO.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF", "BLANK"]

    df_omim = MRCONSO[["CUI", "SAB", "CODE", "STR"]].filter(pl.col("SAB").str.contains("OMIM"))
    df_msh = MRCONSO[["CUI", "SAB", "CODE", "STR"]].filter(pl.col("SAB").str.contains("MSH"))

    df_omim.columns = ["CUI", "OMIM_SAB", "OMIM_CODE", "OMIM_STR"]
    df_msh.columns = ["CUI", "MSH_SAB", "MSH_CODE", "MSH_STR"]

    omim_to_cui = dict(zip(list(df_omim["OMIM_CODE"]), list(df_omim["CUI"])))

    mesh_to_cui = dict(zip(list(df_msh["MSH_CODE"]), list(df_msh["CUI"])))

    os.makedirs(os.path.join(processed_dataset_output), exist_ok=True)

    if not os.path.exists(os.path.join(processed_dataset_output, 'semeval', 'train_semeval_cui.conll')):
        semeval_to_conll(semeval_path, processed_dataset_output)
    
    if not os.path.exists(os.path.join(processed_dataset_output, 'ncbi', 'train_ncbi_cui.conll')):
        ncbi_to_conll(processed_dataset_output, omim_to_cui, mesh_to_cui)

    if not os.path.exists(os.path.join(processed_dataset_output, 'bc5dr', 'train_bc5dr_cui.conll')):
        bc5dr_to_conll(processed_dataset_output, omim_to_cui, mesh_to_cui)

    # turns generated notes to conll with different filtering and modes
    for generated_note_path, dataset in zip(generated_note_paths, datasets):
        if not os.path.exists(os.path.join(processed_dataset_output, f"{dataset}_filteredgen", 'totalfilteredgen.conll')):
            generated_to_conll(generated_note_path, processed_dataset_output, True, mode, dataset)


    models_output = ['BioClinical-ModernBERT-base', 'bert-base-uncased', 'ModernBERT-base']
    models_to_train = ['thomas-sounack/BioClinical-ModernBERT-base', 'google-bert/bert-base-uncased', 'answerdotai/ModernBERT-base']
    
    for dataset in datasets:

        if os.path.exists(results_output):

            num_previous_completed_runs = len([x for x in os.listdir(results_output) if dataset in x and "_ood.csv" in x])
        
        else:
            num_previous_completed_runs = 0
        
        for seed in range(num_previous_completed_runs, 5):

            random_seed = random.randint(0, 2**32 - 2)

            set_seed(random_seed)

            print(f"Run {seed} {random_seed}")

            # run the baseline models
            print(f"training baseline {dataset}")
            baseline_model_performance = train_baseline_berts(processed_dataset_output, results_output, dataset, model_output, models_to_train, models_output)

            print(f"labelling baseline with {dataset} baseline model")
            label_generated_conll(processed_dataset_output, dataset, model_output, models_output)

            if not mixed_training:
                generated_model_performance = train_baseline_generated_berts(processed_dataset_output, results_output, model_output, dataset, False, models_to_train, models_output)
                
                finetuned_model_performance = finetune_berts(processed_dataset_output, results_output, model_output, dataset, models_output)

                performance = baseline_model_performance+generated_model_performance+finetuned_model_performance
                
                with open(os.path.join(results_output, f"{dataset}_mode_{mode}_run_number_{seed}.csv"), "w") as f:
                    csv_out = csv.writer(f)

                    csv_out.writerow(["filename", "precision", "recall", "f1", "accuracy"])

                    for item in performance: 
                        csv_out.writerow(item)
                
                ood_analysis(dataset, processed_dataset_output, results_output, seed, mode)
            
            else:

                tokens, labels = combine_with_generated(processed_dataset_output, f'{processed_dataset_output}/{dataset}_filteredgenlabelled/filteredgeneratedbiobertlabel{dataset}.conll', dataset)

                with open(f'{processed_dataset_output}/{dataset}_filteredgenlabelled/filteredgeneratedbiobertlabel{dataset}_combine.conll', "w") as f:
                    for token, label in zip(tokens, labels):
                        output_string = f'{token.replace(" ", "")} {label}'

                        output_string = output_string.replace("\n", "").strip()

                        f.write(f'{output_string}\n')
                
                tokens, labels = combine_with_generated(processed_dataset_output, f'{processed_dataset_output}/{dataset}_filteredgen/totalfilteredgen.conll', dataset)

                with open(f'{processed_dataset_output}/{dataset}_filteredgen/totalfilteredgen_combine.conll', "w") as f:
                    for token, label in zip(tokens, labels):

                        output_string = f'{token.replace(" ", "")} {label}'

                        output_string = output_string.replace("\n", "").strip()

                        f.write(f'{output_string}\n')

                generated_model_performance = train_baseline_generated_berts(processed_dataset_output, results_output, model_output, dataset, True, models_to_train, models_output)

                performance = baseline_model_performance+generated_model_performance

                with open(os.path.join(results_output, f"{dataset}_mode_{mode}_run_number_{seed}.csv"), "w") as f:
                    csv_out = csv.writer(f)

                    csv_out.writerow(["filename", "precision", "recall", "f1", "accuracy"])

                    for item in performance: 
                        csv_out.writerow(item)

                ood_analysis(dataset, processed_dataset_output, results_output, seed, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)





