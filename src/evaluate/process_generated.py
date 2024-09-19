import csv

import os

import re

from typing import Tuple, List, Optional

from joblib import Parallel, delayed

import pandas as pd

import spacy
from spacy.util import filter_spans

from enum import Enum
 
class Mode(Enum):
    NORMAL = 0
    CHEATING = 1
    REAL_WORLD = 2
    ABLATION = 3

def label_line(text: str, filtered: bool) -> Optional[List[Tuple[int, int, str, str]]]:
    #function to get the nertags for generated text
    nerTags = []

    #removing useless tokens with spaces, take alot of space
    #finding the delimeters for the disease
    re_matches = re.finditer(r'(?<=<1CUI>).*?(?=<\/1CUI>)', text)
    #creating the NERTags
    for match in re_matches:
        nerTags.append((match.start(), match.end(), match.group(0), 'DISEASE'))

    #if its filtered then if the line doesnt have 1CUI tags it will not be added to the final output
    if filtered and len(nerTags)<1:
        return None
    else:
        return nerTags


def process_line(text: str, nlp, filtered: bool) -> Tuple[List[str], List[str]]:
    #processes one line of generated text
    text = text.replace('</s>', ' ').replace('<unk>', ' ').replace("<1CUI>", " <1CUI> ").replace("</1CUI>", " </1CUI> ")
    nerTags = label_line(text, filtered)
    tokens = []
    labels = []
    if nerTags is None:
        return tokens, labels
    
    #labels the spacy doc with the offsets
    doc = nlp(text)
    doc.ents = filter_spans([doc.char_span(s[0], s[1], s[3], alignment_mode="expand") for s in nerTags])

    #gets the tokens and labels from spacy doc
    for sent in doc.sents:
        tokens_sent = [x.text for x in sent]
        iob_tags = [f"{t.ent_type_}" if t.ent_iob_!='O' else "O" for t in sent]

        tokens.extend(tokens_sent)
        labels.extend(iob_tags)

        tokens.extend(' ')
        labels.extend(' ')
    
    indexes = []
    for i in range(len(tokens)):
        if '1CUI' in tokens[i] or (bool(re.search(r"\s{1,}", tokens[i])) and labels[i] != ' ') or "<" in tokens[i] or ">" in tokens[i]:
            indexes.append(i)
    
    for index in sorted(indexes, reverse=True):
        del tokens[index]
        del labels[index]
    
    return tokens, labels

#helper function to divide up work for parallel processing
def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def process_chunk(chunk, nlp, filtered):
    #loop to process one chunk for parallel processing
    tokens = []
    labels = []
    for text in chunk:
        line_tokens, line_labels = process_line(text, nlp, filtered)
        tokens.extend(line_tokens)
        labels.extend(line_labels)
    return tokens, labels


def preprocess_parallel(texts, nlp, filtered, chunksize=100):
    #wrapper function that uses joblib to process all the generated lines
    executor = Parallel(n_jobs=7, verbose=1, max_nbytes=None)
    tasks = (delayed(process_chunk)(chunk, nlp, filtered) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    total_tokens = [] 
    total_labels = []
    for pair in result:
        total_tokens.extend(pair[0])
        total_labels.extend(pair[1])
    return total_tokens, total_labels

def generated_to_conll(generated_output: str, output_path: str, filtered: bool, mode: int, dataset: str) -> None:
    #reads csv of generated text
    df = pd.read_csv(generated_output)

    df = df[~df['matched_output'].isna()]

    if mode == Mode.CHEATING.value: # respectfully cheating
        dataset_test = pd.read_csv(os.path.join(output_path, f"{dataset}/test_{dataset}_cui.conll"), delim_whitespace=True, header = None, names = ["token", "label", "cui"], quoting=csv.QUOTE_NONE)

        test_cuis = set(dataset_test.cui[(dataset_test.cui != "NOCUI") & (dataset_test.cui != "CUI-less")].unique())

        df = df[df['cui'].isin(test_cuis)]
    
    elif mode == Mode.REAL_WORLD.value:
        dataset_train = pd.read_csv(os.path.join(output_path, f"{dataset}/train_{dataset}_cui.conll"), delim_whitespace=True, header = None, names = ["token", "label", "cui"], quoting=csv.QUOTE_NONE)

        train_cuis = set(dataset_train.cui[(dataset_train.cui != "NOCUI") & (dataset_train.cui != "CUI-less")].unique())

        df = df[~df['cui'].isin(train_cuis)]
    
    elif mode == Mode.ABLATION.value:
        dataset_test = pd.read_csv(os.path.join(output_path, f"{dataset}/test_{dataset}_cui.conll"), delim_whitespace=True, header = None, names = ["token", "label", "cui"], quoting=csv.QUOTE_NONE)

        test_cuis = set(dataset_test.cui[(dataset_test.cui != "NOCUI") & (dataset_test.cui != "CUI-less")].unique())

        df = df[~df['cui'].isin(test_cuis)]
    
    #cleans up the data 
    nlp = spacy.load("en_core_sci_md")
    texts = list(df['matched_output'])
    
    #preprocesses all the generated text to tokens and labels
    tokens,labels = preprocess_parallel(texts,nlp, filtered, chunksize=100)

    #writes out the output 
    if filtered:

        filtered_path = os.path.join(output_path, f"{dataset}_filteredgen")

        if not os.path.exists(filtered_path):
            os.mkdir(filtered_path)
        with open(os.path.join(filtered_path, 'totalfilteredgen.conll'), 'w') as f:
            for i in range(len(tokens)):
                f.write(f'{tokens[i]} {labels[i]}\n')

    else:
        
        complete_path = os.path.join(output_path,f"{dataset}_completegen")

        if not os.path.exists(complete_path):
            os.mkdir(complete_path)
        with open(os.path.join(filtered_path, 'totalcompletegen.conll'), 'w') as f:
            for i in range(len(tokens)):
                f.write(f'{tokens[i]} {labels[i]}\n')