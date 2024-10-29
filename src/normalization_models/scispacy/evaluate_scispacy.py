import argparse

import os 

import spacy

import mysql.connector as mariadb

import pandas as pd

from sklearn.metrics import accuracy_score

from scispacy.linking import EntityLinker

from bc5dr import BC5CDR
from ncbi import NCBI
from semeval import Semeval

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

def run_evaluation(data, nlp, omim_to_cui, mesh_to_cui):

    predictions = []

    labels = []

    for row in data:
        doc = nlp(row[0])

        for ent in doc.ents:
            if len(ent._.kb_ents) > 0:
                predictions.append(ent._.kb_ents[0][0])
                labels.append(row[1])
        
            else:
                predictions.append("")
                labels.append(row[1])

    return accuracy_score(labels, map_to_cuis(predictions, omim_to_cui, mesh_to_cui))

def main(args):
    bc5dr_tuples = BC5CDR().generate_data("test")

    ncbi_tuples = NCBI().generate_data("test")

    semeval_tuples = Semeval(args.semeval_data_path).generate_data("test")

    nlp = spacy.load("en_core_sci_sm")

    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

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

    mapped_bc5dr_cuis = map_to_cuis([x[1] for x in bc5dr_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(bc5dr_tuples)):
        bc5dr_tuples[i][1] = mapped_bc5dr_cuis[i]

    mapped_ncbi_cuis = map_to_cuis([x[1] for x in ncbi_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(ncbi_tuples)):
        ncbi_tuples[i][1] = mapped_ncbi_cuis[i]

    mapped_semeval_cuis = map_to_cuis([x[1] for x in semeval_tuples], omim_to_cui, mesh_to_cui)

    for i in range(len(semeval_tuples)):
        semeval_tuples[i][1] = mapped_semeval_cuis[i]

    print("BCDR5 Disease", run_evaluation(bc5dr_tuples, nlp, omim_to_cui, mesh_to_cui))
    print("NCBI", run_evaluation(ncbi_tuples, nlp, omim_to_cui, mesh_to_cui))
    print("Semeval", run_evaluation(semeval_tuples, nlp, omim_to_cui, mesh_to_cui))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("semeval_data_path")
    args = parser.parse_args()

    main(args)