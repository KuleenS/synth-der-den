import csv

import os

from typing import List

import pandas as pd

import numpy as np

import mysql.connector as mariadb

from datasets import load_dataset

from tqdm import tqdm


CUI_FIX = {"C1883552":"C0004093",
"C0021311":"C0009450",
"C0342243":"C0020456",
"C0242362":"C0021818",
"C0228070":"C0027757",
"C1704437":"C0035220",
"C2919362":"C0376670",
"C0050252":"C0377562",
"C0339663":"C0549122",
"C2712054":"C0743330",
"C0557875":"C0849970",
"C0032305":"C1535939",
"C0337197":"C1579887",
"C0005001":"C1704272",
"C1960564":"C2020625",
"C0432747":"C2108211",
"C0151706":"C2979982",
"C0175677":"C3263723",
"C0232280":"C4049827",
"C1883552.":"C0004093",
"C00018946":"C0018946",
"C0040053.":"C0040053",
"CUi-less":"cui-less",
"CUI-less":"cui-less",
"Cui-less":"cui-less",
"C0003878":"C0003873",
"C002076":"C0020676",
"C00240066":"C0240066",
"C023529":"C0235329",
"C03203358":"C3203358"}

class SynonymReplacement:

    def __init__(self, dataset: str, output_file: str, semeval_folder: str = None, mrconso_file: str = None, history_file: str = None) -> None:
        
        self.dataset = dataset

        self.output_file = output_file

        self.semeval_folder = semeval_folder

        self.history_file = history_file

        if self.history_file is not None:
            self.history_df = pd.read_csv(history_file, sep="|", header=None)
        else:
            self.history_df = None

        if mrconso_file is not None:
            mrconso_file = pd.read_csv(mrconso_file, sep="|", header=None)

            df = mrconso_file[[0,11,13,14]]

        else:

            user = os.environ["UMLS_USER"]
            pwd = os.environ["UMLS_PWD"]
            database = os.environ["UMLS_DATABASE_NAME"]
            ip = os.environ["UMLS_IP"]

            mariadb_connection = mariadb.connect(
                user=user, password=pwd, database=database, host=ip
            )

            df = pd.read_sql(
                "SELECT CUI, SAB, CODE, STR FROM MRCONSO",
                con=mariadb_connection,
            )

        df.columns = ["CUI", "SAB", "CODE", "STR"]

        self.code_to_cui = dict(zip(list(df["CODE"]), list(df["CUI"])))

        self.cui_to_strings = dict()

        for _, row in df.iterrows():
            cui_row = row["CUI"]
            cui_str = row["STR"]

            if cui_row not in self.cui_to_strings:

                str_set = set()

                str_set.add(cui_str)
            
            else:
                str_set: set = self.cui_to_strings[cui_row]

                str_set.add(cui_str)
            
            self.cui_to_strings[cui_row] = str_set
    
    def _split_offsets(self, string_input: str) -> np.ndarray:
        return np.array([x.split('-') for x in string_input.split(',')]).astype(int)
    
    def _get_syn(self, cui: str) -> List[str]:
        if cui in self.cui_to_strings:
            return list(self.cui_to_strings[cui])
        elif self.history_df is not None:
            data = list(self.history_df[self.history_df[0] == cui][14])

            if len(data) == 0:
                raise ValueError(f"{cui} does not exist in history")
            else:
                return data

        else:
            raise ValueError(f"{cui} does not exist in cui to string")

    def _syn_replace(self, text: str, start: int, end: int, syns: List[str]) -> List[str]:

        replaced_texts = [text]

        for syn in syns:

            new_text = replaced_texts[0][:start] + f' <1CUI> ' + syn + f' </1CUI> ' + replaced_texts[0][end:]
            
            replaced_texts.append(new_text)

        return replaced_texts
    
    def _map_to_cui(self, item):
        if item in self.code_to_cui:
            return self.code_to_cui[item]
        elif self.history_df is not None:
            data = self.history_df[self.history_df[10] == item][0]

            if len(data) == 0:
                raise ValueError(f"{item} does not exist in history")
            else:
                return data.iloc[0]

        else:
            raise ValueError(f"{item} does not exist in omim/mesh to cui")
    
    def generate(self):
        if self.dataset == "bc5cdr":
            self._process_bc5cdr()
        elif self.dataset == "ncbi":
            self._process_ncbi()
        else:
            self._process_semeval(self.semeval_folder)
    
    def _process_bc5cdr(self):

        dataset = load_dataset("bigbio/bc5cdr")

        passages = dataset["train"]["passages"]

        number_of_cuis = 0

        with open(self.output_file, "w") as f:

            csv_file = csv.writer(f)

            csv_file.writerow(["cui", "matched_output"])

            for passage in tqdm(passages):
                text = passage[0]["text"]+ " " + passage[1]["text"]

                entities = passage[0]["entities"]

                for entity in entities:
                    if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                        offsets = entity["offsets"]

                        normalized = entity["normalized"]

                        for offset, normalized_item in zip(offsets, normalized):

                            mapped_cui = self._map_to_cui(normalized_item["db_id"])

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

            for title, abstract, list_of_mentions in tqdm(zip(titles, abstracts, mentions)):
                text = title + " " + abstract

                for mention in list_of_mentions:
                    if "+" in mention["concept_id"]:
                        concept_ids_split = mention["concept_id"].split("+")
                    else:
                        concept_ids_split = mention["concept_id"].split("|")

                    offset = mention["offsets"]

                    for concept_id_split in concept_ids_split:
                        if "OMIM" in concept_id_split:
                            concept_id_split = concept_id_split.replace("OMIM:", "")
                        
                        concept_id_split = concept_id_split.strip()

                        mapped_cui = self._map_to_cui(concept_id_split)

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

            for filename in tqdm(list_of_pipe_files):
                df = pd.read_csv(filename, sep='|', header=None, on_bad_lines='warn')

                df[2] = df[2].fillna('cui-less')

                df[2] = df[2].replace(CUI_FIX)

                df = df[df[2] != "cui-less"]

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