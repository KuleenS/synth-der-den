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
    REAL_WORLD_PLUS = 3

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

def generated_to_conll(generated_output: str, output_path: str, filtered: bool, mode: int) -> None:
    #reads csv of generated text
    df = pd.read_csv(generated_output)

    df = df[~df['matched_output'].isna()]

    if mode == Mode.CHEATING.value: # respectfully cheating
        semeval_test = pd.read_csv(os.path.join(output_path, "semeval/test_semeval_cui.conll"), sep="\s", header = None, names = ["token", "label", "cui"])

        test_cuis = set(semeval_test.cui[(semeval_test.cui != "NOCUI") & (semeval_test.cui != "CUI-less")].unique())

        df = df[df['cui'].isin(test_cuis)]
    
    elif mode == Mode.REAL_WORLD.value:
        semeval_train = pd.read_csv(os.path.join(output_path, "semeval/train_semeval_cui.conll"), sep="\s", header = None, names = ["token", "label", "cui"])

        train_cuis = set(semeval_train.cui[(semeval_train.cui != "NOCUI") & (semeval_train.cui != "CUI-less")].unique())

        df = df[~df['cui'].isin(train_cuis)]
    
    elif mode == Mode.REAL_WORLD_PLUS.value:
        semeval_test = pd.read_csv(os.path.join(output_path, "semeval/test_semeval_cui.conll"), sep="\s", header = None, names = ["token", "label", "cui"])

        test_cuis = set(semeval_test.cui[(semeval_test.cui != "NOCUI") & (semeval_test.cui != "CUI-less")].unique())

        semeval_test = pd.read_csv(os.path.join(output_path, "semeval/train_semeval_cui.conll"), sep="\s", header = None, names = ["token", "label", "cui"])

        train_cuis = set(semeval_test.cui[(semeval_test.cui != "NOCUI") & (semeval_test.cui != "CUI-less")].unique())

        total_cuis = train_cuis | test_cuis

        df = df[~df['cui'].isin(total_cuis)]

    #cleans up the data 
    nlp = spacy.load("en_core_sci_md")
    texts = list(df['matched_output'])
    
    #preprocesses all the generated text to tokens and labels
    tokens,labels = preprocess_parallel(texts,nlp, filtered, chunksize=100)

    #writes out the output 
    if filtered:

        filtered_path = os.path.join(output_path,'filteredgen')

        if not os.path.exists(filtered_path):
            os.mkdir(filtered_path)
        with open(os.path.join(filtered_path, 'totalfilteredgen.conll'), 'w') as f:
            for i in range(len(tokens)):
                f.write(f'{tokens[i]} {labels[i]}\n')

    else:
        
        complete_path = os.path.join(output_path,'completegen')

        if not os.path.exists(complete_path):
            os.mkdir(complete_path)
        with open(os.path.join(filtered_path, 'totalcompletegen.conll'), 'w') as f:
            for i in range(len(tokens)):
                f.write(f'{tokens[i]} {labels[i]}\n')