import torch
import numpy as np
import pdb
import sys
import wandb
import pickle
from jointmodel_train import train_X_to_C_to_y


def get_info_on_train_test_val_pkl():
    with open('/home/anjilabudathoki/dip-project/project-2025/src/datasets/CUB_processed/class_attr_data_10/train.pkl', 'rb') as f:
        data = pickle.load(f)
        print(f"Type of loaded data: {type(data)}")

        if hasattr(data, '__len__'):
            print(f"Train data: {len(data)}")
        else:
            print("Loaded object does not have a length (not a container type).")
            
    with open('/home/anjilabudathoki/dip-project/project-2025/src/datasets/CUB_processed/class_attr_data_10/val.pkl', 'rb') as f:
        data = pickle.load(f)
        print(f"Type of loaded data: {type(data)}")

        if hasattr(data, '__len__'):
            print(f"Number of elements: {len(data)}")
        else:
            print("Loaded object does not have a length (not a container type).")\
                
    with open('/home/anjilabudathoki/dip-project/project-2025/src/datasets/CUB_processed/class_attr_data_10/test.pkl', 'rb') as f:
        data = pickle.load(f)
        print(f"Type of loaded data: {type(data)}")
        if hasattr(data, '__len__'):
            print(f"Number of elements: {len(data)}")
        else:
            print("Loaded object does not have a length (not a container type).")
    

def run_experiments(dataset, args):
    arg = args[0]
    config_dict = vars(arg)
    experiment = args[0].exp
  
    wandb.init(project="img_processing", config=config_dict, name=f"{experiment}_{arg.seed}_{arg.augmentation_type}_{arg.attr_loss_weight}")
    wandb.config.update(arg)
    train_X_to_C_to_y(*args)



def parse_arguments():
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, 'You need to specify dataset and experiment'
    assert sys.argv[1].upper() in ['CUB'], 'Please specify OAI or CUB dataset'
    assert sys.argv[2] in ['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                           'Standard', 'StandardWithAuxC', 'Multitask', 'Joint', 'Probe',
                           'TTI', 'Robustness', 'HyperparameterSearch'], \
        'Please specify valid experiment. Current: %s' % sys.argv[2]
    dataset = sys.argv[1].upper()
    experiment = sys.argv[2].upper()

    # Handle accordingly to dataset
    from train import parse_arguments

    args = parse_arguments(experiment=experiment)
    return dataset, args

if __name__ == '__main__':
    dataset, args = parse_arguments()  # Parsing the script argument
    np.random.seed(args[0].seed)
    torch.manual_seed(args[0].seed) 

    run_experiments(dataset, args)
