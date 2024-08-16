import os

from typing import List, Tuple

import numpy as np

import pandas as pd

class Semeval:
    def __init__(self, input_folder):
        self.input_folder = input_folder 
    
    def split_offsets(self, string_input: str) -> np.ndarray:
        return np.array([x.split('-') for x in string_input.split(',')]).astype(int)
    
    def generate_data(self, split: str ="train") -> List[Tuple[str, str, str]]:
        list_of_pipe_files = [os.path.join(self.input_folder, split, x)
                                for x in os.listdir(os.path.join(self.input_folder, split)) if x.find('pipe') >= 0]

        results = []

        for filename in list_of_pipe_files:
            df = pd.read_csv(filename, sep='|', header=None, on_bad_lines='warn')
            df[2] = df[2].fillna('CUI-less')

            df = df[df[2] != "CUI-less"]

            if len(df) != 0:

                with open(os.path.join(self.input_folder, split, df[0].iloc[0])) as f:
                    text_document = f.read()

                cuis = list(df[2])
                offsets = list(df[1])

                for cui, offset_group in zip(cuis, offsets):

                    split_offset_group = self.split_offsets(offset_group)

                    for offset in split_offset_group:

                        mention = text_document[offset[0]:offset[1]].strip().replace("\n", " ").replace("\t", " ")

                        results.append([mention, cui, "MESH"])

        return results