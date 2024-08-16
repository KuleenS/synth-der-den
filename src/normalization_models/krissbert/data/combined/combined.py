import torch

from src.normalization_models.krissbert.data.contextual_mention import ContextualMention

class CombinedMentionsDataset(torch.utils.data.Dataset):

    def __init__(self, synthetic, real) -> None:
        super().__init__()
        self.synthetic = synthetic
        self.real = real

        self.docs = self.real.docs + self.synthetic.docs

        self.mentions = self.real.mentions + self.synthetic.mentions

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