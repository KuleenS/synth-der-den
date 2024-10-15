from dataclasses import dataclass 

from typing import List

import torch

from torch import Tensor as T

from transformers import PreTrainedTokenizer

@dataclass
class ContextualMention:
    mention: str
    cuis: List[str]
    ctx_l: str
    ctx_r: str
    start: int
    end: int

    def to_tensor(self, tokenizer: PreTrainedTokenizer, max_length: int) -> T:
        ctx_l_ids = tokenizer.encode(
            text=self.ctx_l,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        ctx_r_ids = tokenizer.encode(
            text=self.ctx_r,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        mention_ids = tokenizer.encode(
            text=self.mention,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        # Concatenate context and mention to the max length.
        token_ids = tokenizer.convert_tokens_to_ids(['<ENT>']) + mention_ids \
            + tokenizer.convert_tokens_to_ids(['</ENT>'])
        max_ctx_len = max_length - len(token_ids) - 2   # Exclude [CLS] and [SEP]
        max_ctx_l_len = max_ctx_len // 2
        max_ctx_r_len = max_ctx_len - max_ctx_l_len
        if len(ctx_l_ids) < max_ctx_l_len and len(ctx_r_ids) < max_ctx_r_len:
            token_ids = ctx_l_ids + token_ids + ctx_r_ids
        elif len(ctx_l_ids) >= max_ctx_l_len and len(ctx_r_ids) >= max_ctx_r_len:
            token_ids = ctx_l_ids[-max_ctx_l_len:] + token_ids \
                + ctx_r_ids[:max_ctx_r_len]
        elif len(ctx_l_ids) >= max_ctx_l_len:
            ctx_l_len = max_ctx_len - len(ctx_r_ids)
            token_ids = ctx_l_ids[-ctx_l_len:] + token_ids + ctx_r_ids
        else:
            ctx_r_len = max_ctx_len - len(ctx_l_ids)
            token_ids = ctx_l_ids + token_ids + ctx_r_ids[:ctx_r_len]

        token_ids = [tokenizer.cls_token_id] + token_ids

        # The above snippet doesn't guarantee the max length limit.
        token_ids = token_ids[:max_length - 1] + [tokenizer.sep_token_id]

        if len(token_ids) < max_length:
            token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))

        return torch.tensor(token_ids)