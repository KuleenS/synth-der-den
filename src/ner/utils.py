from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, tokenizer, test_sentences, test_labels):
        self.tokenizer = tokenizer
        self.test_sentences = test_sentences
        self.test_labels = test_labels

    def __len__(self):
        return len(self.test_sentences)

    def __getitem__(self, idx):
        tokenized_inputs = self.tokenizer(
            self.test_sentences[idx], truncation=True, is_split_into_words=True, max_length=510, padding=False
        )

        word_ids = tokenized_inputs.word_ids(0)

        aligned_labels = align_labels_with_tokens(self.test_labels[idx], word_ids)

        word_ids = [-100 if v is None else v for v in word_ids]

        tokenized_inputs["labels"] = aligned_labels

        tokenized_inputs["word_ids"] = word_ids

        return tokenized_inputs

class NERDatasetCUI(Dataset):
    def __init__(self, tokenizer, test_sentences, test_labels, test_cuis):
        self.tokenizer = tokenizer
        self.test_sentences = test_sentences
        self.test_labels = test_labels
        self.test_cuis = test_cuis


    def __len__(self):
        return len(self.test_sentences)

    def __getitem__(self, idx):
        tokenized_inputs = self.tokenizer(
            self.test_sentences[idx], truncation=True, is_split_into_words=True, max_length=510, padding=False
        )

        word_ids = tokenized_inputs.word_ids(0)

        aligned_labels, aligned_cuis = align_labels_and_cuis_with_tokens(self.test_labels[idx], self.test_cuis[idx], word_ids)

        word_ids = [-100 if v is None else v for v in word_ids]

        tokenized_inputs["labels"] = aligned_labels

        tokenized_inputs["word_ids"] = word_ids

        tokenized_inputs["cuis"] = aligned_cuis

        return tokenized_inputs


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            new_labels.append(label)

    return new_labels

def align_labels_and_cuis_with_tokens(labels, cuis, word_ids):
    new_labels = []
    new_cuis = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            cui = 0 if word_id is None else cuis[word_id]
            new_labels.append(label)
            new_cuis.append(cui)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
            new_cuis.append(0)
        else:
            # Same word as previous token
            label = labels[word_id]
            cui = cuis[word_id]
            # If the label is B-XXX we change it to I-XXX
            new_labels.append(label)
            new_cuis.append(cui)

    return new_labels, new_cuis

def postprocess(predictions, labels, all_labels):

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[all_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions 

def postprocess_cuis(predictions, labels, all_labels):

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[all_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions




