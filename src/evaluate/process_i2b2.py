import os

import re

from typing import List, Tuple

import numpy as np

import spacy

from spacy.util import filter_spans

def process_i2b2(concept_path: str, txt_path: str, concept_path_list: List[str], nlp) -> Tuple[List[str], List[str]]:
    final_tokens = []
    final_labels = []
    for i in range(len(concept_path_list)):
        concept_data = open(os.path.join(concept_path, concept_path_list[i]), 'r').readlines()
        concept_data = [x.replace('||', ' ').replace('\n','') for x in concept_data]

        txt_data = open(os.path.join(txt_path, f"{concept_path_list[i].split('.')[0]}.txt"), 'r').readlines()
        txt_data = [x.replace('\n','').split() for x in txt_data]

        offsets = []
        for line in concept_data:
            text_search = re.search('c=\"(.*)\"\s', line)
            label_search = re.search('t=\"(.*)\"', line)

            label = label_search.group()[3:-1]

            begin = text_search.span()[1]
            end = label_search.span()[0]

            offset = [x.split(":") for x in line[begin:end].split()]
            offset = [[int(y) for y in x] for x in offset]

            [x.append(label) for x in offset]

            offsets.append(offset)

        offsets = sorted(offsets, key=lambda x: x[0][0])

        labels = []

        for line in txt_data:
            labels.append(['O']*len(line))
        
        for offset in offsets:
            line_number = offset[0][0]-1
            begin = offset[0][1]
            end = offset[1][1]
            label = offset[0][2]
            line = txt_data[line_number]
            for i in range(begin, end+1):
                if label=='problem':
                    labels[line_number][i] = 'DISEASE'
        
        flat_txt = [item for sublist in txt_data for item in sublist]
        flat_labels = [item for sublist in labels for item in sublist]

        label_boundaries = np.cumsum([len(x)+1 for x in flat_txt])

        label_boundary_tuples = []

        for i in range(len(label_boundaries)):
            if i == 0:
                label_boundary_tuples.append((0, label_boundaries[i] - 1, flat_labels[i]))
            else:
                label_boundary_tuples.append((label_boundaries[i-1], label_boundaries[i] -1, flat_labels[i]))
        
        text_input = " ".join(flat_txt)

        doc = nlp(text_input)

        #remove overlaps and have clean offsets for labelling of the spacy doct
        doc.ents = filter_spans([doc.char_span(s[0], s[1], label=s[2], alignment_mode="expand") for s in label_boundary_tuples])

        #get the labelled tokens from spacy doc  
        for sent in doc.sents:

            tokens_sent = [x.text for x in sent]
            iob_tags = [f"{t.ent_type_}" if t.ent_iob_!='O' else "O" for t in sent]

            final_tokens.extend(tokens_sent)
            final_labels.extend(iob_tags)

            final_tokens.extend(' ')
            final_labels.extend(' ')

    return final_tokens, final_labels

def i2b2_to_conll(base_path: str, output_path: str):
    concept_path = os.path.join(base_path, 'concept')

    txt_path = os.path.join(base_path, 'txt')

    concept_path_list = sorted(os.listdir(concept_path))

    split_1 = int(0.8 * len(concept_path_list))
    split_2 = int(0.9 * len(concept_path_list))

    train_filenames = concept_path_list[:split_1]

    dev_filenames = concept_path_list[split_1:split_2]

    test_filenames = concept_path_list[split_2:]

    nlp = spacy.load("en_core_sci_md")

    train_tokens, train_labels = process_i2b2(concept_path, txt_path, train_filenames, nlp)

    dev_tokens, dev_labels = process_i2b2(concept_path, txt_path, dev_filenames, nlp)

    test_tokens, test_labels = process_i2b2(concept_path, txt_path, test_filenames, nlp)
    
    output_i2b2_path = os.path.join(output_path, 'i2b2')

    os.makedirs(output_i2b2_path, exist_ok=True)

    with open(os.path.join(output_i2b2_path, 'train_i2b2.conll'), 'w') as f:
        for token, label in zip(train_tokens, train_labels):
            f.write(f'{token} {label}\n')

    with open(os.path.join(output_i2b2_path, 'dev_i2b2.conll'), 'w') as f:
        for token, label in zip(dev_tokens, dev_labels):
            f.write(f'{token} {label}\n')

    with open(os.path.join(output_i2b2_path, 'test_i2b2.conll'), 'w') as f:
        for token, label in zip(test_tokens, test_labels):
            f.write(f'{token} {label}\n')