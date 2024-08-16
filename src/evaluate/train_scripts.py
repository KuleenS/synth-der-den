import os

from src.ner.train import main as train

from argparse import Namespace

def train_baseline_berts(dataset_folder: str, results_folder: str, training_dataset: str, model_output: str) -> None:
    #runs the baseline berts and outputs their results
    models_to_train = ['dmis-lab/biobert-base-cased-v1.1']
    models_output = ['biobert']
    save_directories = [os.path.join(model_output, x) for x in [f'baselines/{training_dataset}/biobert/']]

    for i in range(len(models_to_train)):
        print(f"Training baselines {training_dataset} testing on semeval with {models_to_train[i]}")
        input_files = [
                    os.path.join(dataset_folder, f"{training_dataset}", f"train_{training_dataset}.conll"), 
                    os.path.join(dataset_folder, f"semeval", f"dev_semeval.conll"),
                    os.path.join(dataset_folder, f"semeval", f"test_semeval_cui.conll")]

        output_file = os.path.join(results_folder, f"{models_output[i]}_baseline_trainedon_{training_dataset}_testedwith_semeval.conll")

        args = Namespace(
            files = input_files,
            model_checkpoint = models_to_train[i],
            model_output = save_directories[i],
            conll_output = output_file,
        )

        train(args = args)


def train_baseline_generated_berts(dataset_folder: str, results_folder: str, model_output: str, test_set: str) -> None:
    datasets = [
        'filteredgenlabelled/filteredgeneratedbiobertlabelcleani2b2.conll', 
        'filteredgenlabelled/filteredgeneratedbiobertlabelcleansemeval.conll',
        'filteredgen/totalfilteredgen.conll',
    ]

    datasets = [os.path.join(dataset_folder, x) for x in datasets]

    models_to_train = ['dmis-lab/biobert-base-cased-v1.1']
    models_output = ['biobert']

    save_directories = ["baselines/generated_i2b2_labelled/biobert/", "baselines/generated_semeval_labelled/biobert/", "baselines/generated/biobert/"]

    save_directories = [os.path.join(model_output, x) for x in save_directories]


    for i in range(len(datasets)):
        for j in range(len(models_to_train)):
            
            dataset_path = os.path.basename(datasets[i]).split(".")[0]

            print(f"Training baselines generated {dataset_path} testing on {test_set} with {models_to_train[j]}")

            input_files = [
                    datasets[i], 
                    os.path.join(dataset_folder, f"{test_set}", f"dev_{test_set}.conll"),
                    os.path.join(dataset_folder, f"{test_set}", f"test_{test_set}_cui.conll")
            ]
            
            output_file = os.path.join(results_folder, f"{models_output[j]}_trained_on_{dataset_path}_tested_on_{test_set}.conll")

            args = Namespace(
                files = input_files,
                model_checkpoint = models_to_train[j],
                model_output = save_directories[i],
                conll_output = output_file,
            )

            train(args = args)
    
def finetune_berts(dataset_folder: str, results_folder: str, model_output: str, training_set: str) -> None:
    input_models = ["baselines/generated_i2b2_labelled/biobert/", "baselines/generated_semeval_labelled/biobert/", "baselines/generated/biobert/"]

    output_models = [os.path.join(model_output, "combined", x.split("/")[1]+ f"_{training_set}") for x in input_models]

    input_models = [os.path.join(model_output, x) for x in input_models]

    input_model_names = ["generated_i2b2_labelled", "generated_semeval_labelled", "generated"]

    input_files = [
                    os.path.join(dataset_folder, f"{training_set}" , f"train_{training_set}.conll"), 
                    os.path.join(dataset_folder, f"semeval", f"dev_semeval.conll"),
                    os.path.join(dataset_folder, f"semeval", f"test_semeval_cui.conll")]

    for i in range(len(input_models)):

        output_file = os.path.join(results_folder, f"{input_model_names[i]}_model_then_trained_on_{training_set}_tested_on_semeval.conll")

        print(f"Training baselines generated {training_set} testing on semeval with {input_models[i]}")

        args = Namespace(
                files = input_files,
                model_checkpoint = input_models[i],
                model_output = output_models[i],
                conll_output = output_file,
        )

        train(args = args)