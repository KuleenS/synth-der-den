import torch

from src.normalization_models.krissbert.data.contextual_mention import ContextualMention
from src.normalization_models.krissbert.data.semeval.synthetic_utils import synthetic_process

class SyntheticMentionsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path: str, semeval_data: str, mode: int) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.mode = mode
        self.semeval_data = semeval_data
        self.docs = synthetic_process(
            input_file=self.dataset_path,
            semeval_data=self.semeval_data,
            mode=self.mode,
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