import os

from typing import List

from src.ner.train import main as train

from argparse import Namespace

def train_baseline_berts(dataset_folder: str, results_folder: str, training_dataset: str, model_output: str, models_to_train: List[str], models_output: List[str]) -> List[float]:
    #runs the baseline berts and outputs their results

    model_output_stubs = [f'baselines/{training_dataset}/{x}/' for x in models_output]

    save_directories = [os.path.join(model_output, x) for x in model_output_stubs]

    # thomas-sounack/BioClinical-ModernBERT-large
    # google-bert/bert-large-uncased
    # dmis-lab/biobert-large-cased-v1.1

    model_outputs = []

    for i in range(len(models_to_train)):
        print(f"Training baselines {training_dataset} testing on {training_dataset} with {models_to_train[i]}")
        input_files = [
                    os.path.join(dataset_folder, f"{training_dataset}", f"train_{training_dataset}.conll"), 
                    os.path.join(dataset_folder, f"{training_dataset}", f"dev_{training_dataset}.conll"),
                    os.path.join(dataset_folder, f"{training_dataset}", f"test_{training_dataset}_cui.conll")]

        output_file = os.path.join(results_folder, f"{models_output[i]}_baseline_trained_on_{training_dataset}_testedwith_{training_dataset}.conll")

        os.makedirs(results_folder, exist_ok=True)
        os.makedirs(save_directories[i], exist_ok=True)

        args = Namespace(
            files = input_files,
            model_checkpoint = models_to_train[i],
            model_output = save_directories[i],
            conll_output = output_file,
        )

        model_output = train(args = args)

        model_outputs.append([output_file]+model_output)
    
    return model_outputs


def train_baseline_generated_berts(dataset_folder: str, results_folder: str, model_output: str, test_set: str, combine: bool, models_to_train: List[str], models_output: List[str]) -> List[float]:

    if combine:
        datasets = [
            f'{test_set}_filteredgenlabelled/filteredgeneratedbiobertlabel{test_set}_combine.conll', 
            f'{test_set}_filteredgen/totalfilteredgen_combine.conll',
        ]
    else:
        datasets = [
            f'{test_set}_filteredgenlabelled/filteredgeneratedbiobertlabel{test_set}.conll', 
            f'{test_set}_filteredgen/totalfilteredgen.conll',
        ]


    datasets = [os.path.join(dataset_folder, x) for x in datasets]

    model_output_stubs = [[f"baselines/generated_{test_set}_labelled/{x}/", f"baselines/{test_set}_generated/{x}/"] for x in models_output]

    save_directories = [[os.path.join(model_output, x[0]), os.path.join(model_output, x[1])] for x in model_output_stubs]

    model_outputs = []

    for i in range(len(models_to_train)):
        for j in range(len(datasets)):
            dataset_path = os.path.basename(datasets[j]).split(".")[0]

            print(f"Training baselines generated {dataset_path} testing on {test_set} with {models_to_train[i]}")

            input_files = [
                    datasets[j], 
                    os.path.join(dataset_folder, f"{test_set}", f"dev_{test_set}.conll"),
                    os.path.join(dataset_folder, f"{test_set}", f"test_{test_set}_cui.conll")
            ]
            
            output_file = os.path.join(results_folder, f"{models_output[j]}_trained_on_{dataset_path}_tested_on_{test_set}.conll")

            os.makedirs(results_folder, exist_ok=True)
            os.makedirs(save_directories[i][j], exist_ok=True)

            args = Namespace(
                files = input_files,
                model_checkpoint = models_to_train[i],
                model_output = save_directories[i][j],
                conll_output = output_file,
            )

            model_output = train(args = args)

            model_outputs.append([output_file] + model_output)
    
    return model_outputs
    
def finetune_berts(dataset_folder: str, results_folder: str, model_output: str, training_set: str, models_output: List[str]) -> List[float]:

    model_output_stubs = [[f"baselines/generated_{training_set}_labelled/{x}/", f"baselines/{training_set}_generated/{x}/"] for x in models_output]

    input_models = [[os.path.join(model_output, x[0]), os.path.join(model_output, x[1])] for x in model_output_stubs]

    output_models = [[os.path.join(model_output, "combined", x[0].split("/")[2] + "_"  + x[0].split("/")[1]+ f"_{training_set}"), os.path.join(model_output, "combined", x[1].split("/")[2] + "_"  + x[1].split("/")[1]+ f"_{training_set}")] for x in model_output_stubs]

    input_model_names = [f"generated_{training_set}_labelled", f"{training_set}_generated"]

    input_files = [
                    os.path.join(dataset_folder, f"{training_set}" , f"train_{training_set}.conll"), 
                    os.path.join(dataset_folder, f"{training_set}", f"dev_{training_set}.conll"),
                    os.path.join(dataset_folder, f"{training_set}", f"test_{training_set}_cui.conll")]

    model_outputs = []

    for i in range(len(input_models)):
        for j in range(len(input_models[i])):

            output_file = os.path.join(results_folder, f"{input_model_names[j]}_model_then_trained_on_{training_set}_tested_on_{training_set}.conll")

            print(f"Training baselines generated {training_set} testing on {training_set} with {input_models[i][j]}")

            os.makedirs(results_folder, exist_ok=True)
            os.makedirs(output_models[i][j], exist_ok=True)

            args = Namespace(
                    files = input_files,
                    model_checkpoint = input_models[i][j],
                    model_output = output_models[i][j],
                    conll_output = output_file,
            )

            model_output = train(args = args)

            model_outputs.append([output_file] + model_output)
    
    return model_outputs

