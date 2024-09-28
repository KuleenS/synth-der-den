from typing import Dict, List, Tuple

import torch

from src.normalization_models.krissbert.data.contextual_mention import ContextualMention
from src.normalization_models.krissbert.data.conll.conll_utils import conll_utils_process

class CONLLDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: List[Tuple[str, str]]) -> None:
        super().__init__()

        self.docs = conll_utils_process(dataset)

        self.mentions = []

        self.name_to_cuis = {}

        self._post_init()

    def _post_init(self):
        for d in self.docs:
            self.mentions.extend(d.to_contextual_mentions())
        for m in self.mentions:
            if m.mention not in self.name_to_cuis:
                self.name_to_cuis[m.mention] = set()
            self.name_to_cuis[m.mention].update(m.cuis)

    def __getitem__(self, index: int) -> ContextualMention:
        return self.mentions[index]

    def __len__(self) -> int:
        return len(self.mentions)