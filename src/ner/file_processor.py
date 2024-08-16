import csv

import spacy

import pandas as pd

import numpy as np

from intervaltree import IntervalTree

from typing import List, Tuple

from tqdm.auto import tqdm

class FileProcessor:
    def __init__(self):
        pass
    
    def process_test_cui_file(self, filepath: str)-> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        tokens = []
        labels = []
        cuis = []

        with open(filepath, "r") as f:
            conll_file = [x.strip().replace('" " " "', "") for x in f.readlines()]
        
        sentences = list(self.split(conll_file))

        for sentence in sentences:

            split_sentence = [x.split() for x in sentence]

            tokens_sentence = [x[0] if len(x) == 3 else " " for x in split_sentence]

            labels_sentence = [x[1] if len(x) == 3 else "O" for x in split_sentence]

            cuis_sentence = [x[2] if len(x) == 3 else "NOCUI" for x in split_sentence]

            tokens.append(tokens_sentence)
            labels.append(labels_sentence)
            cuis.append(cuis_sentence)
                
        return tokens, labels, cuis

    def split(self, sequence, sep=''):
        chunk = []
        for val in sequence:
            if val == sep:
                yield chunk
                chunk = []
            else:
                chunk.append(val)
        yield chunk
    
    def process_file(self, filepath: str, test=False) -> Tuple[List[List[str]], List[List[str]]]:
        with open(filepath, "r") as f:
            conll_file = [x.strip().replace('" " " "', "") for x in f.readlines()]
        
        sentences = list(self.split(conll_file))

        tokens = []
        labels = []

        for sentence in sentences:

            split_sentence = [x.split() for x in sentence]

            tokens_sentence = [x[0] if len(x) == 2 else " " for x in split_sentence]
            
            if test:
                labels_sentence = ['O']*len(split_sentence)
            else:
                labels_sentence = [x[1] if len(x) == 2 else "O" for x in split_sentence]

            tokens.append(tokens_sentence)
            labels.append(labels_sentence)
                
        return tokens, labels