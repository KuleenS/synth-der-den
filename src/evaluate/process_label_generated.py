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
    dataset = pd.read_csv(os.path.join(output_data, f'{training_dataset}', f'train_{training_dataset}.conll'), sep='\s', header=None, names=['token','label'], engine='python')
    generated = pd.read_csv(generated_path, sep='\s', header=None, names=["token", "label"])

    stack = pd.concat([dataset, generated],ignore_index=True)
    return stack


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
                model_checkpoint = os.path.join(model_output, "baselines", f'{training_dataset}', f'{models[j]}/'),
                conll_output = output_file,
            )
            
            predict(args=args)


        data = process_labeled_generated_conll(os.path.join(dataset_folder, f"{training_dataset}_{datasets_mode}genlabelled", f"{datasets_mode}generated{models[j]}label{training_dataset}.conll"))
        
        data.to_csv(os.path.join(dataset_folder,f'{training_dataset}_{datasets_mode}genlabelled', f'{datasets_mode}generated{models[j]}labelclean{training_dataset}.conll'), index=False, sep=" ", header=False)

