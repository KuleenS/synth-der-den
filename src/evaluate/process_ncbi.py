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


def process_ncbi(dataset, split, nlp, omim_to_cui: Dict[str, str], mesh_to_cui: Dict[str, str]):
    tokens = []
    labels = []
    cuis = []

    mentions = dataset[split]["mentions"]

    titles = dataset[split]["title"]

    abstracts = dataset[split]["abstract"]

    for title, abstract, list_of_mentions in zip(titles, abstracts, mentions):
        text = title + " " + abstract

        doc = nlp(text)

        doc_spans = []

        for mention in list_of_mentions:
            concept_ids_split = mention["concept_id"].split("|")

            offset = mention["offsets"]

            for concept_id_split in concept_ids_split:
                if "OMIM" in concept_id_split:
                    concept_id_split = concept_id_split.replace("OMIM:", "")
                
                cui = map_to_cui(concept_id_split, omim_to_cui, mesh_to_cui)

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


def ncbi_to_conll(output_path: str, omim_to_cui: Dict[str, str], mesh_to_cui: Dict[str, str]):
    
    nlp = spacy.load("en_core_sci_md")

    dataset = load_dataset("bigbio/ncbi_disease")

    train_tokens, train_labels, train_cuis = process_ncbi(dataset, "train", nlp, omim_to_cui, mesh_to_cui)

    dev_tokens, dev_labels, dev_cuis = process_ncbi(dataset, "validation", nlp, omim_to_cui, mesh_to_cui)

    test_tokens, test_labels, test_cuis = process_ncbi(dataset, "test", nlp, omim_to_cui, mesh_to_cui)

    if not os.path.exists(os.path.join(output_path,'ncbi')):
        os.mkdir(os.path.join(output_path,'ncbi'))

    #writes them out
    with open(os.path.join(output_path, 'ncbi', 'train_ncbi_cui.conll'), 'w') as f:
        for token, label, cui in zip(train_tokens, train_labels, train_cuis):
            f.write(f'{token} {label} {cui}\n')
    with open(os.path.join(output_path, 'ncbi', 'dev_ncbi_cui.conll'), 'w') as f:
        for token, label, cui in zip(dev_tokens, dev_labels, dev_cuis):
            f.write(f'{token} {label} {cui}\n')
    with open(os.path.join(output_path, 'ncbi', 'test_ncbi_cui.conll'), 'w') as f:
        for token, label, cui in zip(test_tokens, test_labels, test_cuis):
            f.write(f'{token} {label} {cui}\n')
    
    with open(os.path.join(output_path, 'ncbi', 'train_ncbi.conll'), 'w') as f:
        for token, label, cui in zip(train_tokens, train_labels, train_cuis):
            f.write(f'{token} {label}\n')
    with open(os.path.join(output_path, 'ncbi', 'dev_ncbi.conll'), 'w') as f:
        for token, label, cui in zip(dev_tokens, dev_labels, dev_cuis):
            f.write(f'{token} {label}\n')
    with open(os.path.join(output_path, 'ncbi', 'test_ncbi.conll'), 'w') as f:
        for token, label, cui in zip(test_tokens, test_labels, test_cuis):
            f.write(f'{token} {label}\n')







   