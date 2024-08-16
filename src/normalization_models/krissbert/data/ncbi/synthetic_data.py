from typing import Dict

from datasets import load_dataset

import torch

from src.normalization_models.krissbert.data.contextual_mention import ContextualMention
from src.normalization_models.krissbert.data.ncbi.synthetic_utils import synthetic_process

class SyntheticNCBIDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path: str, mode: int, omim_to_cui: Dict[str, str], mesh_to_cui: Dict[str, str]) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.mode = mode

        self.omim_to_cui = omim_to_cui
        self.mesh_to_cui = mesh_to_cui

        self.dataset = load_dataset("bigbio/ncbi_disease")

        self.docs = synthetic_process(
            input_file=self.dataset_path,
            dataset=self.dataset,
            mode=self.mode,
            omim_to_cui = self.omim_to_cui,
            mesh_to_cui = self.mesh_to_cui
        )

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