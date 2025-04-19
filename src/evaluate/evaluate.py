import argparse

import csv

import os

import random

import tomli

import pandas as pd

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

    user = os.environ["UMLS_USER"]
    pwd = os.environ["UMLS_PWD"]
    database = os.environ["UMLS_DATABASE_NAME"]
    ip = os.environ["UMLS_IP"]

    mariadb_connection = mariadb.connect(
        user=user, password=pwd, database=database, host=ip
    )

    df_omim = pd.read_sql(
        "SELECT CUI, SAB, CODE, STR FROM MRCONSO WHERE SAB LIKE '%OMIM%'",
        con=mariadb_connection,
    )
    df_msh = pd.read_sql(
        "SELECT CUI, SAB, CODE, STR FROM MRCONSO WHERE SAB LIKE '%MSH%'",
        con=mariadb_connection,
    )

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
    
    for i in range(len(datasets)):

        num_previous_completed_runs = len([x for x in os.listdir(results_output) if datasets[i] in x and "_ood.csv" in x])

        for seed in range(num_previous_completed_runs, 5):

            random_seed = random.randint(0, 2**32 - 2)

            set_seed(random_seed)

            print(f"Run {seed} {random_seed}")

            # run the baseline models
            print(f"training baseline {datasets[i]}")
            baseline_model_performance = train_baseline_berts(processed_dataset_output, results_output, datasets[i], model_output)

            print(f"labelling baseline with {datasets[i]} baseline model")
            label_generated_conll(processed_dataset_output, datasets[i], model_output)

            if not mixed_training:
                generated_model_performance = train_baseline_generated_berts(processed_dataset_output, results_output, model_output, datasets[i], False)
                
                finetuned_model_performance = finetune_berts(processed_dataset_output, results_output, model_output, datasets[i])

                performance = baseline_model_performance+generated_model_performance+finetuned_model_performance
                
                with open(os.path.join(results_output, f"{dataset}_mode_{mode}_run_number_{seed}.csv"), "w") as f:
                    csv_out = csv.writer(f)

                    csv_out.writerow(["filename", "precision", "recall", "f1", "accuracy"])

                    for item in performance: 
                        csv_out.writerow(item)
                
                ood_analysis(datasets[i], processed_dataset_output, results_output, seed, mode)
            
            else:

                tokens, labels = combine_with_generated(processed_dataset_output, f'{processed_dataset_output}/{datasets[i]}_filteredgenlabelled/filteredgeneratedbiobertlabel{datasets[i]}.conll', datasets[i])

                with open(f'{processed_dataset_output}/{datasets[i]}_filteredgenlabelled/filteredgeneratedbiobertlabel{datasets[i]}_combine.conll', "w") as f:
                    for token, label in zip(tokens, labels):
                        f.write(f'{token.replace(" ", "")} {label}\n')
                
                tokens, labels = combine_with_generated(processed_dataset_output, f'{processed_dataset_output}/{datasets[i]}_filteredgen/totalfilteredgen.conll', datasets[i])

                with open(f'{processed_dataset_output}/{datasets[i]}_filteredgen/totalfilteredgen_combine.conll', "w") as f:
                    for token, label in zip(tokens, labels):
                        f.write(f'{token.replace(" ", "")} {label}\n')

                generated_model_performance = train_baseline_generated_berts(processed_dataset_output, results_output, model_output, datasets[i], True)

                performance = baseline_model_performance+generated_model_performance

                with open(os.path.join(results_output, f"{dataset}_mode_{mode}_run_number_{seed}.csv"), "w") as f:
                    csv_out = csv.writer(f)

                    csv_out.writerow(["filename", "precision", "recall", "f1", "accuracy"])

                    for item in performance: 
                        csv_out.writerow(item)

                ood_analysis(datasets[i], processed_dataset_output, results_output, seed, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)





