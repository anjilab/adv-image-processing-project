import torch
import numpy as np
import pdb
import sys
import wandb
import pickle


def run_experiments(dataset, args):
    from train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            train_probe,
            test_time_intervention,
            robustness,
            hyperparameter_optimization,
            adv_attack
        )
    
    arg = args[0]
    config_dict = vars(arg)
    experiment = args[0].exp
  
    # wandb.init(project="img_processing", config=config_dict, name=f"{experiment}_{arg.seed}_{arg.augmentation_type}")
    # wandb.config.update(arg)
    
    # Load the pickle file
    with open('/home/anjilabudathoki/dip-project/project-2025/src/datasets/CUB_processed/class_attr_data_10/train.pkl', 'rb') as f:
        data = pickle.load(f)

        # Display the type and number of elements
        print(f"Type of loaded data: {type(data)}")

        # If it's a list, dictionary, or similar container, get the number of elements
        if hasattr(data, '__len__'):
            print(f"Number of elements: {len(data)}")
        else:
            print("Loaded object does not have a length (not a container type).")


    exit()
    
    if experiment == 'Concept_XtoC':
        # args = args[0]
        
        train_X_to_C(*args) # Model is trained to map the input features to the concept space. 

    elif experiment == 'Independent_CtoY':
        train_oracle_C_to_y_and_test_on_Chat(*args)

    elif experiment == 'Sequential_CtoY':
        train_Chat_to_y_and_test_on_Chat(*args)

    elif experiment == 'Joint':
        train_X_to_C_to_y(*args)

    elif experiment == 'Standard':
        train_X_to_y(*args)


    elif experiment == 'Multitask':
        train_X_to_Cy(*args)

    elif experiment == 'Probe':
        train_probe(*args)

    elif experiment == 'TTI':
        test_time_intervention(*args)

    elif experiment == 'Robustness':
        robustness(*args)
        

    elif experiment == 'HyperparameterSearch':
        hyperparameter_optimization(*args)
        
    elif experiment == 'adv_attack':
        adv_attack(*args)
        

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

    

    dataset, args = parse_arguments() 

    # Seeds
    np.random.seed(args[0].seed)
    torch.manual_seed(args[0].seed)

    run_experiments(dataset, args)
