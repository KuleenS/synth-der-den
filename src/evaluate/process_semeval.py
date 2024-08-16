import os

import re

import pandas as pd
import numpy as np

import spacy
from spacy.util import filter_spans

def split_offsets(string_input: str) -> np.ndarray:
    '''
    turns offsets from semeval text to np array
    123-456, 325-611 -> [[123,456], [325,611]]
    '''
    offset_list = np.array([x.split('-') for x in string_input.split(',')]).astype(int)
    return offset_list

def process_semeval(list_of_pipe_files, path, nlp):
    tokens = []
    labels = []
    cuis = []
    #loops through pipefiles
    for filename in list_of_pipe_files:
        #loads pipefiles
        df = pd.read_csv(filename, sep='|', header=None, on_bad_lines='warn')

        #getting all the offsets from the pipefile
        list_of_offsets = list(df[1])
        list_of_cuis = list(df[2].fillna(""))

        #getting text file from the pipefile 
        #df[0][0] is the name of the text file in the pipefile
        text_input = open(os.path.join(path, df[0][0])).read()

        #getting all the entities text from the offsets
        entities = []
        for i in range(len(list_of_offsets)):
            offsets = split_offsets(list_of_offsets[i])
            for j in range(len(offsets)):
                entities.append((offsets[j][0], offsets[j][1], 'DISEASE', list_of_cuis[i]))


        #loads the text from text file corresponding to pipefile
        doc = nlp(text_input)

        #remove overlaps and have clean offsets for labelling of the spacy doct
        doc.ents = filter_spans([doc.char_span(s[0], s[1], label=s[2], kb_id=s[3], alignment_mode="expand") for s in entities])

        #get the labelled tokens from spacy doc  
        for sent in doc.sents:

            tokens_sent = [x.text for x in sent]
            iob_tags = [f"{t.ent_type_}" if t.ent_iob_!='O' else "O" for t in sent]
            cuis_tags = [f"{t.ent_kb_id_}" if t.ent_kb_id_!="" else "NOCUI" for t in sent]

            tokens.extend(tokens_sent)
            labels.extend(iob_tags)
            cuis.extend(cuis_tags)

            tokens.extend(' ')
            labels.extend(' ')
            cuis.extend(' ')

    #remove additional spaces, underliens
    indexes = []
    for i in range(len(tokens)):
        if (bool(re.search(r"\s{1,}", tokens[i])) and labels[i] != ' ') or bool(re.search(r"_{1,}", tokens[i])) or bool(re.search(r"_{2,}", tokens[i])):
            indexes.append(i)
    for index in sorted(indexes, reverse=True):
        del tokens[index]
        del labels[index]
        del cuis[index]
    
    return tokens, labels, cuis

def semeval_to_conll(semeval_path: str, output_path: str) -> None:
    #getting pipe files from dir
    train_path = os.path.join(semeval_path, 'train')
    dev_path = os.path.join(semeval_path, 'dev')
    test_path = os.path.join(semeval_path, 'test')

    train_filenames=[os.path.join(train_path, x) for x in os.listdir(train_path) if x.find('pipe')>=0]
    dev_filenames=[os.path.join(dev_path, x) for x in os.listdir(dev_path) if x.find('pipe')>=0]
    test_filenames=[os.path.join(test_path, x) for x in os.listdir(test_path) if x.find('pipe')>=0]

    nlp = spacy.load("en_core_sci_md")

    if not os.path.exists(os.path.join(output_path,'semeval')):
        os.mkdir(os.path.join(output_path,'semeval'))

    train_tokens, train_labels, train_cuis = process_semeval(train_filenames, train_path, nlp)
    dev_tokens, dev_labels, dev_cuis = process_semeval(dev_filenames, dev_path, nlp)
    test_tokens, test_labels, test_cuis = process_semeval(test_filenames, test_path, nlp)

    #writes them out
    with open(os.path.join(output_path, 'semeval', 'train_semeval_cui.conll'), 'w') as f:
        for token, label, cui in zip(train_tokens, train_labels, train_cuis):
            f.write(f'{token} {label} {cui}\n')
    with open(os.path.join(output_path, 'semeval', 'dev_semeval_cui.conll'), 'w') as f:
        for token, label, cui in zip(dev_tokens, dev_labels, dev_cuis):
            f.write(f'{token} {label} {cui}\n')
    with open(os.path.join(output_path, 'semeval', 'test_semeval_cui.conll'), 'w') as f:
        for token, label, cui in zip(test_tokens, test_labels, test_cuis):
            f.write(f'{token} {label} {cui}\n')
    
    with open(os.path.join(output_path, 'semeval', 'train_semeval.conll'), 'w') as f:
        for token, label, cui in zip(train_tokens, train_labels, train_cuis):
            f.write(f'{token} {label}\n')
    with open(os.path.join(output_path, 'semeval', 'dev_semeval.conll'), 'w') as f:
        for token, label, cui in zip(dev_tokens, dev_labels, dev_cuis):
            f.write(f'{token} {label}\n')
    with open(os.path.join(output_path, 'semeval', 'test_semeval.conll'), 'w') as f:
        for token, label, cui in zip(test_tokens, test_labels, test_cuis):
            f.write(f'{token} {label}\n')