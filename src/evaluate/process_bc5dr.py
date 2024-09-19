import os

from typing import Dict

import spacy
from spacy.util import filter_spans

from datasets import load_dataset


def map_to_cui(item, omim_to_cui, mesh_to_cui):
    if item in omim_to_cui:
        return omim_to_cui[item]
    elif item in mesh_to_cui:
        return mesh_to_cui[item]
    return item


def process_bc5dr(dataset, split, nlp, omim_to_cui: Dict[str, str], mesh_to_cui: Dict[str, str]):
    tokens = []
    labels = []
    cuis = []

    passages = dataset[split]["passages"]

    for passage in passages:

        text_input =  passage[0]["text"]+ " " + passage[1]["text"]

        doc = nlp(text_input)

        doc_spans = []

        entities = passage[0]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                offsets = entity["offsets"]

                normalized = entity["normalized"]

                for offset, normalized_item in zip(offsets, normalized):

                    cui = map_to_cui(normalized_item["db_id"], omim_to_cui, mesh_to_cui)

                    doc_spans.append(doc.char_span(offset[0], offset[1], label="DISEASE", kb_id=cui, alignment_mode="expand"))
        
        entities = passage[1]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                offsets = entity["offsets"]

                normalized = entity["normalized"]

                for offset, normalized_item in zip(offsets, normalized):

                    cui = map_to_cui(normalized_item["db_id"], omim_to_cui, mesh_to_cui)

                    doc_spans.append(doc.char_span(offset[0], offset[1], label="DISEASE", kb_id=cui, alignment_mode="expand"))
        
        doc.ents = filter_spans(doc_spans)

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

    return tokens, labels, cuis


def bc5dr_to_conll(output_path: str, omim_to_cui: Dict[str, str], mesh_to_cui: Dict[str, str]):
    
    nlp = spacy.load("en_core_sci_md")

    dataset = load_dataset("bigbio/bc5cdr")

    train_tokens, train_labels, train_cuis = process_bc5dr(dataset, "train", nlp, omim_to_cui, mesh_to_cui)

    dev_tokens, dev_labels, dev_cuis = process_bc5dr(dataset, "validation", nlp, omim_to_cui, mesh_to_cui)

    test_tokens, test_labels, test_cuis = process_bc5dr(dataset, "test", nlp, omim_to_cui, mesh_to_cui)

    if not os.path.exists(os.path.join(output_path,'bc5dr')):
        os.mkdir(os.path.join(output_path,'bc5dr'))

    #writes them out
    with open(os.path.join(output_path, 'bc5dr', 'train_bc5dr_cui.conll'), 'w') as f:
        for token, label, cui in zip(train_tokens, train_labels, train_cuis):
            f.write(f'{token} {label} {cui}\n')
    with open(os.path.join(output_path, 'bc5dr', 'dev_bc5dr_cui.conll'), 'w') as f:
        for token, label, cui in zip(dev_tokens, dev_labels, dev_cuis):
            f.write(f'{token} {label} {cui}\n')
    with open(os.path.join(output_path, 'bc5dr', 'test_bc5dr_cui.conll'), 'w') as f:
        for token, label, cui in zip(test_tokens, test_labels, test_cuis):
            f.write(f'{token} {label} {cui}\n')
    
    with open(os.path.join(output_path, 'bc5dr', 'train_bc5dr.conll'), 'w') as f:
        for token, label, cui in zip(train_tokens, train_labels, train_cuis):
            f.write(f'{token} {label}\n')
    with open(os.path.join(output_path, 'bc5dr', 'dev_bc5dr.conll'), 'w') as f:
        for token, label, cui in zip(dev_tokens, dev_labels, dev_cuis):
            f.write(f'{token} {label}\n')
    with open(os.path.join(output_path, 'bc5dr', 'test_bc5dr.conll'), 'w') as f:
        for token, label, cui in zip(test_tokens, test_labels, test_cuis):
            f.write(f'{token} {label}\n')







   