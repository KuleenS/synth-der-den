import numpy as np

from src.normalization_models.krissbert.data.document import Document
from src.normalization_models.krissbert.data.mention import Mention

def conll_utils_process(dataset):
    krissbert_dataset = []

    d = Document()

    data = [x for x in dataset if len(x) == 2]

    text = " ".join(x[0] for x in data)

    token_offsets = np.cumsum([len(x[0])+1 for x in data])

    tokens = [x[1] for x in data]

    token_offsets_pairs = []

    for i in range(len(token_offsets) - 1):
        if i == 0:
            token_offsets_pairs.append((0, token_offsets[i]))
        else:
            token_offsets_pairs.append((token_offsets[i], token_offsets[i+1]))

    for token, token_offsets_pair in zip(tokens, token_offsets_pairs):
        if token == "DISEASE":

            m = Mention(cui=None, start=token_offsets_pair[0], end=token_offsets_pair[1], text=text)

            d.mentions.append(m)
            
    krissbert_dataset.append(d)
    
    return krissbert_dataset