import os

import re

import pandas as pd

from src.normalization_models.krissbert.data.document import Document
from src.normalization_models.krissbert.data.mention import Mention

from src.normalization_models.krissbert.utils import Mode

def synthetic_process(input_file: str, semeval_data: str, mode: int):
    dataset = []

    synthetic_df = pd.read_csv(input_file)

    synthetic_df = synthetic_df[~synthetic_df['matched_output'].isna()]

    synthetic_cuis = list(synthetic_df["cui"])

    synthetic_outputs = list(synthetic_df["matched_output"])

    base_semeval_dir = os.path.dirname(semeval_data)

    train_file_dir = os.path.join(base_semeval_dir, "train")

    test_file_dir = os.path.join(base_semeval_dir, "test")

    train_files = [os.path.join(train_file_dir, x)
                              for x in os.listdir(train_file_dir) if x.find('pipe') >= 0]

    test_files =  [os.path.join(test_file_dir, x)
                              for x in os.listdir(test_file_dir) if x.find('pipe') >= 0]

    train_cuis = set()

    test_cuis = set()
    
    for train_file in train_files:

        df = pd.read_csv(train_file, sep='|', header=None, on_bad_lines='warn')

        df[2] = df[2].fillna('CUI-less')

        train_cuis.update(list(df[2]))
    
    for test_file in test_files:

        df = pd.read_csv(test_file, sep='|', header=None, on_bad_lines='warn')

        df[2] = df[2].fillna('CUI-less')

        test_cuis.update(list(df[2]))

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
        
        dataset.append(d)

    return dataset