from enum import Enum

import torch
from transformers import PreTrainedTokenizer

from tqdm import tqdm

class Mode(Enum):
    SEMEVAL = 0
    SEMEVAL_NORMAL = 1
    SEMEVAL_CHEATING = 2
    SEMEVAL_REAL_WORLD = 3
    ABLATION = 4

def generate_vectors(
    encoder: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    max_length: int,
    is_prototype: bool = False,
):
    n = len(dataset)
    total = 0
    results = []
    for i, batch_start in tqdm(enumerate(range(0, n, batch_size))):
        batch = [dataset[i] for i in range(batch_start, min(n, batch_start + batch_size))]
        batch_token_tensors = [m.to_tensor(tokenizer, max_length) for m in batch]

        ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
        seg_batch = torch.zeros_like(ids_batch)
        attn_mask = (ids_batch != tokenizer.pad_token_id)

        with torch.inference_mode():
            out = encoder(
                input_ids=ids_batch,
                token_type_ids=seg_batch,
                attention_mask=attn_mask
            )
            out = out[0][:, 0, :]
        out = out.cpu()

        num_mentions = out.size(0)
        total += num_mentions

        if is_prototype:
            meta_batch = [{'cuis': m.cuis} for m in batch]
            assert len(meta_batch) == num_mentions
            results.extend([(meta_batch[i], out[i].view(-1).numpy()) for i in range(num_mentions)])
        else:
            results.extend(out.cpu().split(1, dim=0))

    if not is_prototype:
        results = torch.cat(results, dim=0)

    return results