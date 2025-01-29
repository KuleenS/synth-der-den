import argparse
import os
import tomli

from llama import LLaMaGenerate
from input_data import InputData

def main(args):

    config_path = args.config
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    generation_config = config['generate']

    output_folder = generation_config['output_folder']

    input_file = generation_config['input_file']

    model_dir = generation_config['model_dir']

    generation_params = generation_config['generation_params']

    decoder_name = generation_params["model_name"].replace("/", "")

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_file):
        input_data = InputData(input_file)

        input_data.generate()

    llama_umls_data = LLaMaGenerate(input_file, os.path.join(output_folder, f"generation_data_{decoder_name}.csv"), model_dir, generation_params)

    llama_umls_data.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)
