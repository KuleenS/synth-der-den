import re

from typing import Dict

import pandas as pd

from src.normalization_models.krissbert.data.document import Document
from src.normalization_models.krissbert.data.mention import Mention

from src.normalization_models.krissbert.utils import Mode

def map_to_cuis(items, omim_to_cui, mesh_to_cui):
    
    new_items = []

    for item in items:
        new_items.append(map_to_cui(item, omim_to_cui, mesh_to_cui))
    
    return new_items

def map_to_cui(item, omim_to_cui, mesh_to_cui):
    if item in omim_to_cui:
        return omim_to_cui[item]
    elif item in mesh_to_cui:
        return mesh_to_cui[item]
    return item

def synthetic_process(input_file: str, dataset, mode: int, omim_to_cui: Dict[str, str], mesh_to_cui: Dict[str, str]):
    total_dataset = []

    synthetic_df = pd.read_csv(input_file)

    synthetic_df = synthetic_df[~synthetic_df['matched_output'].isna()]

    synthetic_cuis = list(synthetic_df["cui"])

    synthetic_outputs = list(synthetic_df["matched_output"])

    train_cuis = []

    train_passages = dataset["train"]["passages"]

    for passage in train_passages:

        entities = passage[0]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:
                
                train_cuis.append(entity["normalized"][0]["db_id"])
        
        entities = passage[1]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:
                train_cuis.append(entity["normalized"][0]["db_id"])
    
    test_cuis = []
    
    test_passages = dataset["test"]["passages"]

    for passage in test_passages:

        entities = passage[0]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:
                
                test_cuis.append(entity["normalized"][0]["db_id"])
        
        entities = passage[1]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:
                test_cuis.append(entity["normalized"][0]["db_id"])
    
    train_cuis = set(map_to_cuis(train_cuis, omim_to_cui, mesh_to_cui))
    test_cuis = set(map_to_cuis(test_cuis, omim_to_cui, mesh_to_cui))

    for cui, output in zip(synthetic_cuis, synthetic_outputs):
        
        d = Document()

        output = output.replace('</s>', ' ').replace('<unk>', ' ').replace("<1CUI>", " <1CUI> ").replace("</1CUI>", " </1CUI> ")

        re_matches = re.finditer(r'(?<=<1CUI>).*?(?=<\/1CUI>)', output)

        for re_match in re_matches:
            m = Mention(cui=cui, start=re_match.start(), end=re_match.end(), text=output)

            if mode == Mode.SEMEVAL_NORMAL.value:
                d.mentions.append(m)

            elif mode == Mode.SEMEVAL_CHEATING.value and cui in test_cuis:
                 d.mentions.append(m)
              
            elif mode == Mode.SEMEVAL_REAL_WORLD.value and cui not in train_cuis:
                 d.mentions.append(m)
                
            elif mode == Mode.ABLATION.value and cui not in test_cuis:
                 d.mentions.append(m)
        
        total_dataset.append(d)

    return total_dataset