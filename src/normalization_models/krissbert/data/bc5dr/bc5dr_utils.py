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

def bc5dr_utils_process(dataset, split, omim_to_cui, mesh_to_cui):
    passages = dataset[split]["passages"]

    krissbert_dataset = []

    for passage in passages:

        d = Document()

        text = passage[0]["text"]+ " " + passage[1]["text"]

        entities = passage[0]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                offsets = entity["offsets"]

                normalized = entity["normalized"]

                for offset, normalized_item in zip(offsets, normalized):

                    m = Mention(cui=map_to_cui(normalized_item["db_id"], omim_to_cui, mesh_to_cui), start=offset[0], end=offset[1], text=text)

                    d.mentions.append(m)
                
        entities = passage[1]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                offsets = entity["offsets"]

                normalized = entity["normalized"]

                for offset, normalized_item in zip(offsets, normalized):

                    m = Mention(cui=map_to_cui(normalized_item["db_id"], omim_to_cui, mesh_to_cui), start=offset[0], end=offset[1], text=text)

                    d.mentions.append(m)
        
        krissbert_dataset.append(d)
    
    return krissbert_dataset


def bc5dr_utils_process_ood(dataset, omim_to_cui, mesh_to_cui):

    train_cuis = []

    train_passages = dataset["train"]["passages"]

    for passage in train_passages:

        entities = passage[0]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:
                
                train_cuis.append(entity["normalized"][0]["db_id"])
        
        entities = passage[1]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:
                train_cuis.append(entity["normalized"][0]["db_id"])
    
    train_cuis = set(map_to_cuis(train_cuis, omim_to_cui, mesh_to_cui))

    passages = dataset["test"]["passages"]

    krissbert_dataset = []

    for passage in passages:

        d = Document()

        text = passage[0]["text"]+ " " + passage[1]["text"]

        entities = passage[0]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                offsets = entity["offsets"]

                normalized = entity["normalized"]

                for offset, normalized_item in zip(offsets, normalized):

                    mapped_cui = map_to_cui(normalized_item["db_id"], omim_to_cui, mesh_to_cui)

                    if mapped_cui not in train_cuis:

                        m = Mention(cui=mapped_cui, start=offset[0], end=offset[1], text=text)

                        d.mentions.append(m)
                
        entities = passage[1]["entities"]

        for entity in entities:
            if entity["type"] == "Disease" and len(entity["normalized"]) != 0:

                offsets = entity["offsets"]

                normalized = entity["normalized"]

                for offset, normalized_item in zip(offsets, normalized):

                    mapped_cui = map_to_cui(normalized_item["db_id"], omim_to_cui, mesh_to_cui)

                    if mapped_cui not in train_cuis:

                        m = Mention(cui=mapped_cui, start=offset[0], end=offset[1], text=text)

                        d.mentions.append(m)
        
        krissbert_dataset.append(d)
    
    return krissbert_dataset