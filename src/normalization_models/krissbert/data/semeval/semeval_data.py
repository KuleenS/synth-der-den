import torch

from src.normalization_models.krissbert.data.contextual_mention import ContextualMention
from src.normalization_models.krissbert.data.semeval.semeval_utils import semeval_process, semeval_process_ood

class SemevalMentionsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path: str, ood: bool = False) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.ood = ood

        if self.ood:
            self.docs = semeval_process_ood(
                input_folder=dataset_path,
            )
        else:
            self.docs = semeval_process(
                input_folder=dataset_path,
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