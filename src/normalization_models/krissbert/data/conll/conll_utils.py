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

    for i in range(len(token_offsets)):
        if i == 0:
            token_offsets_pairs.append((0, token_offsets[i]))
        else:
            token_offsets_pairs.append((token_offsets[i-1], token_offsets[i]))

    j = 0 

    while j < len(tokens):
        if tokens[j] == "DISEASE":
            
            k = 0

            while j+k < len(tokens) and tokens[j+k] == "DISEASE":
                k+=1 

            m = Mention(cui=None, start=token_offsets_pairs[j][0], end=token_offsets_pairs[j+k-1][1], text=text)

            d.mentions.append(m)
                
            j += k
        else:
            j += 1
            
    krissbert_dataset.append(d)
    
    return krissbert_dataset