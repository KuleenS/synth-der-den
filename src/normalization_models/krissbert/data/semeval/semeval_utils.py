import os

import pandas as pd

import numpy as np 

from src.normalization_models.krissbert.data.document import Document

from src.normalization_models.krissbert.data.mention import Mention

def split_offsets(string_input: str) -> np.ndarray:
        return np.array([x.split('-') for x in string_input.split(',')]).astype(int)

def semeval_process(input_folder: str):
    dataset = []

    list_of_pipe_files = [os.path.join(input_folder, x)
                              for x in os.listdir(input_folder) if x.find('pipe') >= 0]


    for filename in list_of_pipe_files:
        df = pd.read_csv(filename, sep='|', header=None, on_bad_lines='warn')
        df[2] = df[2].fillna('CUI-less')

        df = df[df[2] != "CUI-less"]

        if len(df) != 0:

            with open(os.path.join(input_folder, df[0].iloc[0])) as f:
                text_document = f.read()


            d = Document()

            cuis = list(df[2])
            offsets = list(df[1])

            for cui, offset_group in zip(cuis, offsets):

                split_offset_group = split_offsets(offset_group)

                for offset in split_offset_group:
                    m = Mention(cui=cui, start=offset[0], end=offset[1], text=text_document)

                    d.mentions.append(m)
            
            dataset.append(d)

    return dataset

def semeval_process_ood(input_folder: str):
    dataset = []

    list_of_pipe_files = [os.path.join(input_folder, x)
                              for x in os.listdir(input_folder) if x.find('pipe') >= 0]
    
    base_semeval_dir = os.path.dirname(input_folder)
    
    train_file_dir = os.path.join(base_semeval_dir, "train")

    train_files = [os.path.join(train_file_dir, x)
                              for x in os.listdir(train_file_dir) if x.find('pipe') >= 0]

    train_cuis = set()

    for train_file in train_files:

        df = pd.read_csv(train_file, sep='|', header=None, on_bad_lines='warn')

        df[2] = df[2].fillna('CUI-less')

        train_cuis.update(list(df[2]))
    
    for filename in list_of_pipe_files:
        df = pd.read_csv(filename, sep='|', header=None, on_bad_lines='warn')
        df[2] = df[2].fillna('CUI-less')

        df = df[df[2] != "CUI-less"]

        if len(df) != 0:

            with open(os.path.join(input_folder, df[0].iloc[0])) as f:
                text_document = f.read()


            d = Document()

            cuis = list(df[2])
            offsets = list(df[1])

            for cui, offset_group in zip(cuis, offsets):

                if cui not in train_cuis:

                    split_offset_group = split_offsets(offset_group)

                    for offset in split_offset_group:
                        m = Mention(cui=cui, start=offset[0], end=offset[1], text=text_document)

                        d.mentions.append(m)
            
            dataset.append(d)

    return dataset