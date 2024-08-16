import torch

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass

from transformers.data.data_collator import DataCollatorMixin

@dataclass
class DataCollatorForTokenClassificationCustom(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding = True
    max_length = 510
    pad_to_multiple_of = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        word_ids = [feature["word_ids"] for feature in features]

        cuis = None

        if "cuis" in features[0].keys():
            cuis = [feature["cuis"] for feature in features]

        no_labels_features = [{k: v for k, v in feature.items() if k not in [label_name, "word_ids", "cuis"]} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]
        
        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        
        if padding_side == "right":
            batch["word_ids"] = [
                to_list(word_id) + [-100] * (sequence_length - len(word_id)) for word_id in word_ids
            ]
        else:
            batch["word_ids"] = [
                [-100] * (sequence_length - len(word_id)) + to_list(word_id) for word_id in word_ids
            ]
        
        if cuis is not None:
            if padding_side == "right":
                batch["cuis"] = [
                    to_list(cui) + [0] * (sequence_length - len(cui)) for cui in cuis
                ]
            else:
                batch["cuis"] = [
                    [0] * (sequence_length - len(cui)) + to_list(cui) for cui in cuis
                ]
            
            batch["cuis"] = torch.tensor(batch["cuis"], dtype=torch.int64)
        
        batch["word_ids"] = torch.tensor(batch["word_ids"], dtype=torch.int64)

        return batch