import argparse
from train import MolRLTrainFactory, MolRLInferenceFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to config file")
    parser.add_argument("-i", "--inference", action="store_true", help="inference mode")
    args = parser.parse_args()
    config_path = args.config_path
    
    if not args.inference:
        MolRLTrainFactory.from_yaml(config_path) \
            .create_train() \
            .train() \
            .close()
    else:
        MolRLInferenceFactory.from_yaml(config_path) \
            .create_inference() \
            .inference() \
            .close()