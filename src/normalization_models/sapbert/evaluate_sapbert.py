import argparse

import os 

import faiss 

import mysql.connector as mariadb

import numpy as np

import pandas as pd

import torch

from transformers import AutoTokenizer, AutoModel  

from tqdm import tqdm

from src.normalization_models.sapbert.bc5dr import BC5CDR
from src.normalization_models.sapbert.ncbi import NCBI
from src.normalization_models.sapbert.semeval import Semeval
from src.normalization_models.sapbert.synthetic import Synthetic


def accuracy_at_k(predictions, labels, k):

    counter = 0

    for prediction, label in zip(predictions, labels):
        if label in prediction[:k]:
            counter += 1

    return counter/len(predictions)

def run_evaluation_synthetic(train, synthetic_train, test, tokenizer, model):
    bs = 128
    all_reps = []
    cui_index = []

    total_train = train + synthetic_train

    for i in tqdm(np.arange(0, len(total_train), bs)):

        cui_index.extend([x[1] for x in total_train[i:i+bs]])

        batch = [x[0] for x in total_train[i:i+bs]]

        toks = tokenizer(batch, padding=True, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model(**toks)
        
        cls_rep = output[0][:,0,:]
        
        all_reps.append(cls_rep.cpu().detach().numpy())

    train_cuis = set([x[1] for x in train])
    
    all_reps_emb = np.concatenate(all_reps, axis=0)

    index = faiss.IndexFlatL2(all_reps_emb.shape[1])
    index.add(all_reps_emb) 

    predictions = []

    labels = []

    for i in tqdm(np.arange(0, len(test), 64)): 

        batch = [x[0] for x in test[i:i+bs]]

        label = [x[1] for x in test[i:i+bs]]

        toks = tokenizer(batch, padding=True, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model(**toks)

        query_cls_rep = output[0][:,0,:].cpu().detach().numpy()

        _, I = index.search(query_cls_rep, 100)

        for prediction_row in I:
            predictions.append([cui_index[x] for x in prediction_row])
        
        labels.extend(label)
    
    regular_accuracy = accuracy_at_k(predictions, labels, 1), accuracy_at_k(predictions, labels, 5), accuracy_at_k(predictions, labels, 10), accuracy_at_k(predictions, labels, 50), accuracy_at_k(predictions, labels, 100)

    labels_ood, predictions_ood = [], []

    for prediction, label in zip(predictions, labels):
        if label not in train_cuis:
            labels_ood.append(label)
            predictions_ood.append(prediction)

    ood_accuracy = accuracy_at_k(predictions_ood, labels_ood, 1), accuracy_at_k(predictions_ood, labels_ood, 5), accuracy_at_k(predictions_ood, labels_ood, 10), accuracy_at_k(predictions_ood, labels_ood, 50), accuracy_at_k(predictions_ood, labels_ood, 100)

    return regular_accuracy, ood_accuracy

def run_evaluation(train, test, tokenizer, model):
    bs = 128
    all_reps = []
    cui_index = []
    for i in tqdm(np.arange(0, len(train), bs)):

        cui_index.extend([x[1] for x in train[i:i+bs]])

        batch = [x[0] for x in train[i:i+bs]]

        toks = tokenizer(batch, padding=True, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model(**toks)
        
        cls_rep = output[0][:,0,:]
        
        all_reps.append(cls_rep.cpu().detach().numpy())

    train_cuis = set(cui_index)
    
    all_reps_emb = np.concatenate(all_reps, axis=0)

    index = faiss.IndexFlatL2(all_reps_emb.shape[1])
    index.add(all_reps_emb) 

    predictions = []

    labels = []

    for i in tqdm(np.arange(0, len(test), 64)): 

        batch = [x[0] for x in test[i:i+bs]]

        label = [x[1] for x in test[i:i+bs]]

        toks = tokenizer(batch, padding=True, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model(**toks)

        query_cls_rep = output[0][:,0,:].cpu().detach().numpy()

        _, I = index.search(query_cls_rep, 100)

        for prediction_row in I:
            predictions.append([cui_index[x] for x in prediction_row])
        
        labels.extend(label)
    
    regular_accuracy = accuracy_at_k(predictions, labels, 1), accuracy_at_k(predictions, labels, 5), accuracy_at_k(predictions, labels, 10), accuracy_at_k(predictions, labels, 50), accuracy_at_k(predictions, labels, 100)

    labels_ood, predictions_ood = [], []

    for prediction, label in zip(predictions, labels):
        if label not in train_cuis:
            labels_ood.append(label)
            predictions_ood.append(prediction)

    ood_accuracy = accuracy_at_k(predictions_ood, labels_ood, 1), accuracy_at_k(predictions_ood, labels_ood, 5), accuracy_at_k(predictions_ood, labels_ood, 10), accuracy_at_k(predictions_ood, labels_ood, 50), accuracy_at_k(predictions_ood, labels_ood, 100)

    return regular_accuracy, ood_accuracy


def map_to_cuis(items, omim_to_cui, mesh_to_cui):
    
    new_items = []

    for item in items:
        if item in omim_to_cui:
            new_items.append(omim_to_cui[item])
        elif item in mesh_to_cui:
            new_items.append(mesh_to_cui[item])
        else:
            new_items.append(item)
    
    return new_items

def main(args):
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

    train_bc5dr_tuples = BC5CDR().generate_data("train")

    train_ncbi_tuples = NCBI().generate_data("train")

    train_semeval_tuples = Semeval(args.semeval_data_path).generate_data("train")

    train_mapped_bc5dr_cuis = map_to_cuis([x[1] for x in train_bc5dr_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(train_bc5dr_tuples)):
        train_bc5dr_tuples[i][1] = train_mapped_bc5dr_cuis[i]

    train_mapped_ncbi_cuis = map_to_cuis([x[1] for x in train_ncbi_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(train_ncbi_tuples)):
        train_ncbi_tuples[i][1] = train_mapped_ncbi_cuis[i]

    train_mapped_semeval_cuis = map_to_cuis([x[1] for x in train_semeval_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(train_semeval_tuples)):
        train_semeval_tuples[i][1] = train_mapped_semeval_cuis[i]
    
    test_bc5dr_tuples = BC5CDR().generate_data("test")

    test_ncbi_tuples = NCBI().generate_data("test")

    test_semeval_tuples = Semeval(args.semeval_data_path).generate_data("test")

    test_mapped_bc5dr_cuis = map_to_cuis([x[1] for x in test_bc5dr_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(test_bc5dr_tuples)):
        test_bc5dr_tuples[i][1] = test_mapped_bc5dr_cuis[i]

    test_mapped_ncbi_cuis = map_to_cuis([x[1] for x in test_ncbi_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(test_ncbi_tuples)):
        test_ncbi_tuples[i][1] = test_mapped_ncbi_cuis[i]

    test_mapped_semeval_cuis = map_to_cuis([x[1] for x in test_semeval_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(test_semeval_tuples)):
        test_semeval_tuples[i][1] = test_mapped_semeval_cuis[i]

    synthetic_data_generator = Synthetic()

    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to("cuda")

    model.eval()

    print("BCDR5 Disease", run_evaluation(train_bc5dr_tuples, test_bc5dr_tuples, tokenizer, model))
    print("NCBI", run_evaluation(train_ncbi_tuples, test_ncbi_tuples, tokenizer, model))
    print("Semeval", run_evaluation(train_semeval_tuples, test_semeval_tuples, tokenizer, model))

    generated_input = args.synthetic_data

    datasets = [("BC5DR", train_bc5dr_tuples, test_bc5dr_tuples), ("NCBI", train_ncbi_tuples, test_ncbi_tuples), ("Semeval", train_semeval_tuples, test_semeval_tuples)]

    for dataset_name, train, test in datasets:

        mode_1_synthetic = synthetic_data_generator.generate_data(generated_input, train, test, 1)
        mode_2_synthetic = synthetic_data_generator.generate_data(generated_input, train, test, 2)
        mode_3_synthetic = synthetic_data_generator.generate_data(generated_input, train, test, 3)
        mode_4_synthetic = synthetic_data_generator.generate_data(generated_input, train, test, 4)

        synthetic_data = [mode_1_synthetic, mode_2_synthetic, mode_3_synthetic, mode_4_synthetic]

        for i in range(len(synthetic_data)):
            print(f"Mode {i+1}")
            print(dataset_name, run_evaluation_synthetic(train, synthetic_data[i], test, tokenizer, model))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("semeval_data_path")
    parser.add_argument("synthetic_data")

    args = parser.parse_args()

    main(args)