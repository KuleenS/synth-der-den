from src.normalization_models.krissbert.data.document import Document
from src.normalization_models.krissbert.data.mention import Mention

def map_to_cuis(items, omim_to_cui, mesh_to_cui):
    
    new_items = []

    for item in items:
        new_items.append(map_to_cui(item, omim_to_cui, mesh_to_cui))
    
    return new_items

def map_to_cui(item, omim_to_cui, mesh_to_cui):
    if item in omim_to_cui:
        return omim_to_cui[item]
    elif item in mesh_to_cui:
        return mesh_to_cui[item]
    return item


def ncbi_utils_process(dataset, split, omim_to_cui, mesh_to_cui):

    mentions = dataset[split]["mentions"]

    titles = dataset[split]["title"]

    abstracts = dataset[split]["abstract"]

    krissbert_dataset = []

    for title, abstract, list_of_mentions in zip(titles, abstracts, mentions):
        text = title + " " + abstract

        d = Document()

        for mention in list_of_mentions:
            concept_ids_split = mention["concept_id"].split("|")

            offset = mention["offsets"]

            for concept_id_split in concept_ids_split:
                if "OMIM" in concept_id_split:
                    concept_id_split = concept_id_split.replace("OMIM:", "")
                    
                m = Mention(cui=map_to_cui(concept_id_split, omim_to_cui, mesh_to_cui), start=offset[0], end=offset[1], text=text)

                d.mentions.append(m)
        
        krissbert_dataset.append(d)

    return krissbert_dataset                     
        
def ncbi_utils_process_ood(dataset, omim_to_cui, mesh_to_cui):

    mentions = dataset["train"]["mentions"]

    train_cuis = []

    for mention in mentions:
        for mention in mentions:
            for concept_id in mention:
                concept_ids_split = concept_id["concept_id"].split("|")
                for concept_id_split in concept_ids_split:
                    if "OMIM" in concept_id_split:
                        concept_id_split = concept_id_split.replace("OMIM:", "")
                    train_cuis.append(concept_id_split)

    mentions = dataset["test"]["mentions"]

    titles = dataset["test"]["title"]

    abstracts = dataset["test"]["abstract"]

    krissbert_dataset = []

    train_cuis = set(map_to_cuis(train_cuis, omim_to_cui, mesh_to_cui))

    for title, abstract, list_of_mentions in zip(titles, abstracts, mentions):
        text = title + " " + abstract

        d = Document()

        for title, abstract, list_of_mentions in zip(titles, abstracts, mentions):
            text = title + " " + abstract

            d = Document()

            for mention in list_of_mentions:
                concept_ids_split = mention["concept_id"].split("|")

                offset = mention["offsets"]

                for concept_id_split in concept_ids_split:
                    if "OMIM" in concept_id_split:
                        concept_id_split = concept_id_split.replace("OMIM:", "")
                    
                    concept_id_split = map_to_cui(concept_id_split, mesh_to_cui, omim_to_cui)
                    
                    if concept_id_split not in train_cuis:

                        m = Mention(cui=concept_id_split, start=offset[0], end=offset[1], text=text)

                        d.mentions.append(m)
        
        krissbert_dataset.append(d)
    
    return krissbert_dataset

