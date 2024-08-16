import argparse

import os
import pathlib
import pickle

import mysql.connector as mariadb

import pandas as pd

from transformers import AutoConfig, AutoTokenizer, AutoModel

from src.normalization_models.krissbert.data.combined.combined import CombinedMentionsDataset
from src.normalization_models.krissbert.data.semeval.semeval_data import SemevalMentionsDataset
from src.normalization_models.krissbert.data.semeval.synthetic_data import SyntheticMentionsDataset

from src.normalization_models.krissbert.data.bc5dr.bc5dr_data import BCD5DRDataset
from src.normalization_models.krissbert.data.bc5dr.synthetic_data import SyntheticBC5DRDataset

from src.normalization_models.krissbert.data.ncbi.ncbi_data import NCBIDataset
from src.normalization_models.krissbert.data.ncbi.synthetic_data import SyntheticNCBIDataset

from src.normalization_models.krissbert.utils import Mode, generate_vectors

def main(args):

    mode = args.mode

    dataset = args.dataset

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )
    encoder = AutoModel.from_pretrained(
        args.model_name,
        config=config
    )

    encoder.cuda()
    encoder.eval()

    ds = None

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


    if dataset == "semeval":

        if mode == Mode.SEMEVAL.value:

            ds = SemevalMentionsDataset(args.semeval_input)
        
        else:
            print(f"Combining synthetic with {mode} mode")
            real = SemevalMentionsDataset(args.semeval_input)

            synth = SyntheticMentionsDataset(args.generated_input, args.semeval_input, mode=mode)

            ds = CombinedMentionsDataset(synth, real)
    
    elif dataset == "bc5dr":

        if mode == Mode.SEMEVAL.value:

            ds = BCD5DRDataset("train", False, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)
        
        else:
            print(f"Combining synthetic with {mode} mode")
            real = BCD5DRDataset("train", False, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)

            synth = SyntheticBC5DRDataset(args.generated_input, mode=mode, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)

            ds = CombinedMentionsDataset(synth, real)
    
    elif dataset == "ncbi":

        if mode == Mode.SEMEVAL.value:

            ds = NCBIDataset("train", False, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)
        
        else:
            print(f"Combining synthetic with {mode} mode")
            real = NCBIDataset("train", False, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)

            synth = SyntheticNCBIDataset(args.generated_input, mode=mode, omim_to_cui=omim_to_cui, mesh_to_cui=mesh_to_cui)

            ds = CombinedMentionsDataset(synth, real)

    data = generate_vectors(encoder, tokenizer, ds, 4096, 128, is_prototype=True)

    pathlib.Path(os.path.join(args.output, "prototypes")).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.output, "prototypes", "embeddings"), mode="wb") as f:
        pickle.dump(data, f)
    with open(os.path.join(args.output, "prototypes", "cuis"), 'w') as f:
        for name, cuis in ds.name_to_cuis.items():
            f.write('|'.join(cuis) + '||' + name.replace("||", "") + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL")
    parser.add_argument("--output")
    parser.add_argument("--semeval_input", default = None)
    parser.add_argument("--dataset")
    parser.add_argument("--generated_input")
    parser.add_argument("--mode", type=int)


    args = parser.parse_args()

    main(args)