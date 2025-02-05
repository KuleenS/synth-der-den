import csv

import os

from typing import List

import pandas as pd

import numpy as np

import mysql.connector as mariadb

from datasets import load_dataset


class SynonymReplacement:

    def __init__(self, dataset: str, output_file: str, input_folder: str = None) -> None:
        
        self.dataset = dataset

        self.output_file = output_file

        self.input_folder = input_folder

        self.query = "SELECT STR FROM MRCONSO WHERE CUI='{cui}' AND LAT='ENG AND ISPREF = 'Y';"

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

        self.omim_to_cui = dict(zip(list(df_omim["OMIM_CODE"]), list(df_omim["CUI"])))

        self.mesh_to_cui = dict(zip(list(df_msh["MSH_CODE"]), list(df_msh["CUI"])))
    
    def _split_offsets(string_input: str) -> np.ndarray:
        return np.array([x.split('-') for x in string_input.split(',')]).astype(int)
    
    def _get_syn(self, cui: str) -> List[str]:

        syns = pd.read_sql(self.query.format(cui=cui))

        syns.columns = ["STR"]

        return list(syns["STR"])

    def _syn_replace(self, text: str, start: int, end: int, syns: List[str]) -> List[str]:

        replaced_texts = [text]

        for syn in syns:

            new_text = text = text[:start] + f' <1CUI> ' + syn + f' <1CUI> ' + text[end:]
            
            replaced_texts.append(new_text)

        return replaced_texts
    
    def _map_to_cui(item, omim_to_cui, mesh_to_cui):
        if item in omim_to_cui:
            return omim_to_cui[item]
        elif item in mesh_to_cui:
            return mesh_to_cui[item]
        return item
    
    def generate(self):
        if self.dataset == "bc5cdr":
            self._process_bc5cdr()
        elif self.dataset == "ncbi":
            self._process_ncbi()
        else:
            self._process_semeval(self.input_folder)
    
    def _process_bc5cdr(self):

        dataset = load_dataset("bigbio/bc5cdr")

        passages = dataset["train"]["passages"]

        with open(self.output_file, "w") as f:

            csv_file = csv.writer(f)

            csv_file.writerow(["cui", "matched_output"])

            for passage in passages:
                text = passage[0]["text"]+ " " + passage[1]["text"]

                entities = passage[0]["entities"]

                for entity in entities:
                    if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                        offsets = entity["offsets"]

                        normalized = entity["normalized"]

                        for offset, normalized_item in zip(offsets, normalized):

                            mapped_cui = self._map_to_cui(normalized_item["db_id"], self.omim_to_cui, self.mesh_to_cui)

                            synoynms = self._get_syn(mapped_cui)

                            syn_replaced_texts = self._syn_replace(text, offset[0], offset[1], synoynms)

                            for syn_replaced_text in syn_replaced_texts:
                                csv_file.writerow([mapped_cui, syn_replaced_text])

    def _process_ncbi(self):

        dataset = load_dataset("bigbio/ncbi_disease")

        mentions = dataset["train"]["mentions"]

        titles = dataset["train"]["title"]

        abstracts = dataset["train"]["abstract"]

        with open(self.output_file, "w") as f:

            csv_file = csv.writer(f)

            csv_file.writerow(["cui", "matched_output"]) 

            for title, abstract, list_of_mentions in zip(titles, abstracts, mentions):
                text = title + " " + abstract

                for mention in list_of_mentions:
                    concept_ids_split = mention["concept_id"].split("|")

                    offset = mention["offsets"]

                    for concept_id_split in concept_ids_split:
                        if "OMIM" in concept_id_split:
                            concept_id_split = concept_id_split.replace("OMIM:", "")

                        mapped_cui = self._map_to_cui(concept_id_split, self.omim_to_cui, self.mesh_to_cui)

                        synoynms = self._get_syn(mapped_cui)

                        syn_replaced_texts = self._syn_replace(text, offset[0], offset[1], synoynms)
                            
                        for syn_replaced_text in syn_replaced_texts:
                            csv_file.writerow([mapped_cui, syn_replaced_text])

    def _process_semeval(self, input_folder: str):
        list_of_pipe_files = [os.path.join(input_folder, x)
                                for x in os.listdir(input_folder) if x.find('pipe') >= 0]

        with open(self.output_file, "w") as f:

            csv_file = csv.writer(f)

            csv_file.writerow(["cui", "matched_output"]) 

            for filename in list_of_pipe_files:
                df = pd.read_csv(filename, sep='|', header=None, on_bad_lines='warn')
                df[2] = df[2].fillna('CUI-less')

                df = df[df[2] != "CUI-less"]

                if len(df) != 0:

                    with open(os.path.join(input_folder, df[0].iloc[0])) as f:
                        text_document = f.read()

                    cuis = list(df[2])
                    offsets = list(df[1])

                    for cui, offset_group in zip(cuis, offsets):
                        split_offset_group = self._split_offsets(offset_group)

                        for offset in split_offset_group:
                            synoynms = self._get_syn(cui)

                            syn_replaced_texts = self._syn_replace(text_document, offset[0], offset[1], synoynms)
                                
                            for syn_replaced_text in syn_replaced_texts:
                                csv_file.writerow([cui, syn_replaced_text])