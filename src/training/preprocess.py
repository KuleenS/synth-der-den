import os

from typing import List, Tuple

import torch

from transformers import AutoTokenizer

from tqdm import tqdm

import pandas as pd

from src.training.dataset import SupervisedGenerationDataset

class SupervisedDataPreprocess:
    def __init__(self, file_path: str, tokenizer: AutoTokenizer,
                 special_tokens_path: str = None, extra_tokens_path: str = None):

        self.file_path = file_path
        self.special_tokens_path = special_tokens_path
        self.extra_tokens_path = extra_tokens_path
        self.tokenizer = tokenizer
    
    def process_files(self) -> Tuple[List[str], List[str]]:
        files = [os.path.join(self.file_path,x) for x in os.listdir(self.file_path)]
        source = []

        target = []

        for i in range(len(files)):
            df = pd.read_csv(files[i], header=None)
            source.extend(list(df[0]))
            target.extend(list(df[1]))
        
        if self.special_tokens_path is not None:
            replacement_tokens = open(self.special_tokens_path, "r").read().split("\n")
            for i in range(len(target)):
                for j in range(len(replacement_tokens)):
                    target[i] = target[i].replace(replacement_tokens[j], '')
                    source[i] = source[i].replace(replacement_tokens[j], '')
        
        return source, target
    
    def tokenize(self, examples: Tuple[List[str], List[str]]) -> SupervisedGenerationDataset:
        
        if self.extra_tokens_path is not None:
            self.tokenizer.add_tokens(open(self.extra_tokens_path, "r").read().split("\n"))
        
        inputs = []
        input_attention_masks = []
        labels = []

        sources, targets = examples

        for i in tqdm(range(len(sources))):
            
            # tokenize the inputs and target to max_length of size of model
            tokenized_input = self.tokenizer(sources[i], padding='max_length', return_tensors='pt', truncation = True)

            tokenized_target = self.tokenizer(targets[i], padding='max_length', return_tensors='pt', truncation = True)

            inputs.append(tokenized_input['input_ids'])
            input_attention_masks.append(tokenized_input['attention_mask'])
            labels.append(tokenized_target['input_ids'])

        
        inputs = torch.cat(inputs)
        input_attention_masks = torch.cat(input_attention_masks)
        labels = torch.cat(labels)

        dataset = SupervisedGenerationDataset(inputs, input_attention_masks, labels)

        return dataset