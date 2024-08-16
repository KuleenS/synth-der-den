from enum import Enum

import os

import re

from typing import Tuple, List

import pandas as pd

class Mode(Enum):
    SEMEVAL = 0
    SEMEVAL_NORMAL = 1
    SEMEVAL_CHEATING = 2
    SEMEVAL_REAL_WORLD = 3
    SEMEVAL_REAL_WORLD_PLUS = 4

class Synthetic:

    def __init__(self) -> None:
        pass

    def generate_data(self, input_file: str, train_data: List[Tuple[str, str, str]], test_data: List[Tuple[str, str, str]], mode: int) -> List[Tuple[str,str,str]]:

        results = []

        synthetic_df = pd.read_csv(input_file)

        synthetic_df = synthetic_df[~synthetic_df['matched_output'].isna()]

        synthetic_cuis = list(synthetic_df["cui"])

        synthetic_outputs = list(synthetic_df["matched_output"])

        train_cuis = set([x[1] for x in train_data])

        test_cuis = set([x[1] for x in test_data])

        for cui, output in zip(synthetic_cuis, synthetic_outputs):
            output = output.replace('</s>', ' ').replace('<unk>', ' ').replace("<1CUI>", " <1CUI> ").replace("</1CUI>", " </1CUI> ")

            re_matches = re.finditer(r'(?<=<1CUI>).*?(?=<\/1CUI>)', output)

            for re_match in re_matches:
                mention = output[re_match.start():re_match.end()].strip().replace("\n", " ").replace("\t", " ")

                m = [mention, cui, "MESH"]

                if mode == Mode.SEMEVAL_NORMAL.value:
                    results.append(m)

                elif mode == Mode.SEMEVAL_CHEATING.value and cui in test_cuis:
                    results.append(m)
                
                elif mode == Mode.SEMEVAL_REAL_WORLD.value and cui not in train_cuis:
                    results.append(m)
                    
                elif mode == Mode.SEMEVAL_REAL_WORLD_PLUS.value and (cui not in train_cuis and cui not in test_cuis):
                    results.append(m)
            
        return results