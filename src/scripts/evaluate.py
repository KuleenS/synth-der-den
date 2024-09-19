import argparse

import os

import random

import tomli

import pandas as pd

import numpy as np

import torch

import mysql.connector as mariadb

from src.evaluate.process_i2b2 import i2b2_to_conll
from src.evaluate.process_ncbi import ncbi_to_conll
from src.evaluate.process_bc5dr import bc5dr_to_conll
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
    datasets = ['semeval','bc5dr', 'ncbi']

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

    semeval_to_conll(semeval_path, processed_dataset_output)
    i2b2_to_conll(i2b2_base_path, processed_dataset_output)
    ncbi_to_conll(processed_dataset_output, omim_to_cui, mesh_to_cui)
    bc5dr_to_conll(processed_dataset_output, omim_to_cui, mesh_to_cui)

    # turns generated notes to conll with different filtering and modes
    for dataset in datasets:
        for j in range(len(filtering)):
            generated_to_conll(generated_note_path, processed_dataset_output, filtering[j], mode, dataset)
    
    for i in range(len(datasets)):
        for seed in range(5):

            random_seed = random.randint(0, 100_000)

            set_seed(random_seed)

            print(f"Run {seed} {random_seed}")

            # run the baseline models
            for i in range(len(datasets)):
                print(f"training baseline {datasets[i]}")
                train_baseline_berts(processed_dataset_output, results_output, datasets[i], model_output)

            # # labels generated notes with those baseline modes
            
                print(f"labelling baseline with {datasets[i]} baseline model")
                label_generated_conll(processed_dataset_output, datasets[i], model_output)

            train_baseline_generated_berts(processed_dataset_output, results_output, model_output, datasets[i])
            
            for dataset in datasets:
                finetune_berts(processed_dataset_output, results_output, model_output, dataset)
            
            ood_analysis(processed_dataset_output, results_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)





