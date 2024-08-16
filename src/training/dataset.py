from torch.utils.data import Dataset

class SupervisedGenerationDataset(Dataset):
    def __init__(self, inputs, input_attention_masks, labels):
        self.inputs = inputs
        self.input_attention_masks = input_attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):

        return {"input_ids": self.inputs[idx], "attention_mask": self.input_attention_masks[idx], "labels" : self.labels[idx]}