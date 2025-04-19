import os

from argparse import Namespace


from ..ner.predict import main as predict

import pandas as pd


def process_labeled_generated_conll(path: str):
    #helper function to get rid of confidence, gold, and extra columns in labelled text from uabdeid
    data = pd.read_csv(path, sep=' ', header=None, names=["token", "label"])
    data.reset_index(inplace=True)
    data = data[["token", "label"]]
    return data

def combine_with_generated(output_data: str, generated_path: str, training_dataset: str):
    #helper function to combine the generated text with the semeval text
    
    with open(os.path.join(output_data, f'{training_dataset}', f'train_{training_dataset}.conll'), "r") as f:
        training_data = [x.split(" ") for x in f.readlines()]

        training_data = [x if len(x) == 2 else ('', '') for x in training_data ]

        training_tokens, training_labels = zip(*training_data)

    with open(generated_path, "r") as f:
        generated_data = [x.split(" ") for x in f.readlines()]

        generated_data = [x if len(x) == 2 else ('', '') for x in generated_data ]

        generated_tokens, generated_labels = zip(*generated_data)
    
    return training_tokens+ generated_tokens, training_labels + generated_labels

def label_generated_conll(dataset_folder: str, training_dataset: str, model_output: str):
    models = ['biobert']
    datasets_modes = ['filtered']
    os.environ['MKL_THREADING_LAYER'] = 'GNU'


    for i in range(len(datasets_modes)):
        datasets_mode = datasets_modes[i]
        os.makedirs(os.path.join(dataset_folder, f'{training_dataset}_{datasets_mode}genlabelled'), exist_ok=True)
        for j in range(len(models)):

            input_file = os.path.join(dataset_folder, f"{training_dataset}_{datasets_mode}gen", f"total{datasets_mode}gen.conll")

            output_file = os.path.join(dataset_folder, f"{training_dataset}_{datasets_mode}genlabelled", f"{datasets_mode}generated{models[j]}label{training_dataset}.conll")

            args = Namespace(
                file = input_file,
                files = None,
                model_checkpoint = os.path.join(model_output, "baselines", f'{training_dataset}', f'{models[j]}/'),
                conll_output = output_file,
            )
            
            predict(args=args)