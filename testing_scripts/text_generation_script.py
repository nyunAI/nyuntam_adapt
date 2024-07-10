import yaml
import os
import argparse
from tasks import CausalLLM


def run_experiment(config):
    with open(os.path.join(config), "r") as stream:
        data_loaded = yaml.safe_load(stream)
    print(data_loaded)

    train_llama = CausalLLM(**data_loaded)
    train_llama.adapt_model()


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment with a given YAML configuration."
    )
    parser.add_argument("config_file", help="Path to the YAML configuration file")

    args = parser.parse_args()
    config_file = args.config_file

    # Run the experiment
    run_experiment(config_file)


if __name__ == "__main__":
    main()
