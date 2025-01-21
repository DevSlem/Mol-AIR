import argparse
from train import MolRLTrainFactory, MolRLInferenceFactory, MolRLPretrainFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to config file")
    parser.add_argument("-i", "--inference", action="store_true", help="inference mode")
    parser.add_argument("-p", "--pretrain", action="store_true", help="pretrain mode")
    args = parser.parse_args()
    config_path = args.config_path
    
    # Pretraining
    if args.pretrain:
        MolRLPretrainFactory.from_yaml(config_path) \
            .create_pretrain() \
            .pretrain() \
            .close()
    # RL Inference
    elif args.inference:
        MolRLInferenceFactory.from_yaml(config_path) \
            .create_inference() \
            .inference() \
            .close()
    # RL Training
    else:
        MolRLTrainFactory.from_yaml(config_path) \
            .create_train() \
            .train() \
            .close()