import csv

from typing import Dict, Any

import pandas as pd

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig

from tqdm import tqdm 

class LLaMaGenerate:

    def __init__(self, input_file: str, output_file: str, model_dir: str, model_params: Dict[str, Any]) -> None:

        self.input_file = input_file
        self.output_file = output_file
        self.model_dir = model_dir
        self.model_name = model_params['model_name']
        self.batch_size = model_params['batch_size']
        self.gpus = model_params['gpus']
        self.examples_generated = model_params['examples_generated']

        self.gen_config = GenerationConfig.from_pretrained("meta-llama/Llama-2-13b-hf")

        self.gen_config.max_new_tokens = 256
    
    def generate(self):

        df = pd.read_csv(self.input_file, quotechar='"')
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, max_new_tokens=2048)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 2048

        tokenizer.padding_side = "left"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(self.model_dir, quantization_config=quantization_config, device_map="auto")

        model.eval()

        with open(self.output_file, "w") as f:

            out = csv.writer(f)

            out.writerow(["cui", "string", "prompt", "output"])

            for i in tqdm(range(int(len(df)/self.batch_size)-1)):

                batch_output = []
                batch_output_cuis = []
                batch_output_strings = []
                batch_output_prompts = []
                
                batch = df.iloc[i*self.batch_size:(i+1)*self.batch_size]
                batch_prompts, batch_cuis, batch_strings = list(batch['PROMPT']), list(batch['CUI']), list(batch['STR'])

                batch_prompts = [f"### Instruction: {x}\n ### Example Doctor's Note:" for x in batch_prompts]

                tokenized_sentence = tokenizer(batch_prompts, padding=True, return_tensors='pt').to(0)

                for j in range(self.examples_generated):
                    
                    with torch.no_grad():
                        output_ids = model.generate(
                            **tokenized_sentence,

                            generation_config = self.gen_config
                        )

                    decoded = tokenizer.batch_decode(output_ids.cpu())

                    batch_output.extend(decoded)

                    batch_output_cuis.extend(batch_cuis)

                    batch_output_strings.extend(batch_strings)

                    batch_output_prompts.extend(batch_prompts)
                
                out.writerows(list(zip(batch_output_cuis, batch_output_strings, batch_output_prompts, batch_output)))
