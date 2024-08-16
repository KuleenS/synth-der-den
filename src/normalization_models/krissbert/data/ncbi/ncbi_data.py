from typing import Dict

from datasets import load_dataset

import torch

from src.normalization_models.krissbert.data.contextual_mention import ContextualMention
from src.normalization_models.krissbert.data.ncbi.ncbi_utils import ncbi_utils_process, ncbi_utils_process_ood

class NCBIDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, ood: bool, omim_to_cui: Dict[str, str], mesh_to_cui: Dict[str, str]) -> None:
        super().__init__()
        self.dataset = load_dataset("bigbio/ncbi_disease")

        self.ood = ood

        self.omim_to_cui = omim_to_cui
        self.mesh_to_cui = mesh_to_cui

        if self.ood:
            self.docs = ncbi_utils_process_ood(self.dataset, self.omim_to_cui, self.mesh_to_cui)
        else:
            self.docs = ncbi_utils_process(self.dataset, split, self.omim_to_cui, self.mesh_to_cui)

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