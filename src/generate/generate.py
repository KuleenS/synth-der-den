import argparse
import os
import tomli

from llama import LLaMaGenerate
from input_data import InputData
from synonym_replacement import SynonymReplacement
from template import Templating

def main(args):

    config_path = args.config
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    generation_config = config['generate']

    output_folder = generation_config['output_folder']

    for synth_data in generation_config['synthetic_data']:

        if synth_data == "llm":
            input_file = generation_config["llm"]['input_file']

            model_dir = generation_config["llm"]['model_dir']

            generation_params = generation_config["llm"]['generation_params']

            decoder_name = generation_params["model_name"].replace("/", "")

            os.makedirs(output_folder, exist_ok=True)

            if not os.path.exists(input_file):
                input_data = InputData(input_file)

                input_data.generate()
            
            llama_umls_data = LLaMaGenerate(input_file, os.path.join(output_folder, f"generation_data_{decoder_name}.csv"), model_dir, generation_params)

            llama_umls_data.generate()

        elif synth_data == "template":
            template_data = Templating(os.path.join(output_folder, f"template_data.csv"))

            template_data.generate()
        
        elif synth_data == "synonym":

            dataset = generation_config["synonym"]["dataset"]

            semeval_folder = generation_config["synonym"]["semeval_folder"]

            syn_data = SynonymReplacement(dataset, os.path.join(output_folder, f"syn_replacement_data.csv"), semeval_folder)

            syn_data.generate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)
