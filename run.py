import argparse
from train import MolRLTrainFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to config file")
    args = parser.parse_args()
    config_path = args.config_path
    
    MolRLTrainFactory.from_yaml(config_path) \
        .create_train() \
        .train() \
        .close()
