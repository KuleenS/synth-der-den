from dataclasses import dataclass, field

from typing import List

from src.normalization_models.krissbert.data.mention import Mention
from src.normalization_models.krissbert.data.contextual_mention import ContextualMention


@dataclass
class Document:
    mentions: List[Mention] = field(default_factory=list)

    def to_contextual_mentions(self, max_length: int = 64) -> List[ContextualMention]:
        mentions = []
        for m in self.mentions:
            ctx_l, ctx_r = m.text[:m.start].strip().split(), m.text[m.end:].strip().split()
            ctx_l, ctx_r = ' '.join(ctx_l[-max_length:]), ' '.join(ctx_r[:max_length])

            cm = ContextualMention(
                mention=m.text[m.start:m.end].strip().replace("\n", " ").replace("\t", " "),
                cuis=[m.cui],
                ctx_l=ctx_l,
                ctx_r=ctx_r,
            )

            mentions.append(cm)
        
        return mentions