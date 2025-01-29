import argparse
import tomli

from src.preprocess.generate_semeval_data import GenerateSemevalData


def main(args):

    config_path = args.config
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    preprocess_config = config['preprocess']

    semeval_data = preprocess_config['semeval_data']
    output_data = preprocess_config['semeval_output_data']

    GenerateSemevalData(semeval_data, output_data).generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args)
