import torch
import numpy as np
import pdb
import sys
import wandb
import argparse
from config import BASE_DIR, N_ATTRIBUTES

from template_model import MLP, inception_v3, End2EndModel

import pdb
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy

from dataset import load_data, find_class_imbalance
from config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
import wandb



def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            #inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        
        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters
    return loss_meter, acc_meter

if __name__ == '__main__':
    
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, 'You need to specify dataset and experiment'
    assert sys.argv[1].upper() in ['CUB'], 'Please specify OAI or CUB dataset'
    assert sys.argv[2] in ['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                           'Standard', 'StandardWithAuxC', 'Multitask', 'Joint', 'Probe',
                           'TTI', 'Robustness', 'HyperparameterSearch'], \
        'Please specify valid experiment. Current: %s' % sys.argv[2]
    dataset = sys.argv[1].upper()
    experiment = sys.argv[2].upper()
    
     # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Multitask', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch'],
                        help='Name of experiment to run.')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')
    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
    parser.add_argument('-lr', type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                        help='Whether to use weighted loss for single attribute or multiple ones')
    parser.add_argument('-uncertain_labels', action='store_true',
                        help='whether to use (normalized) attribute certainties as labels')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-expand_dim', type=int, default=0,
                        help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
    parser.add_argument('-n_class_attr', type=int, default=2,
                        help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
    parser.add_argument('-end2end', action='store_true',
                        help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
    parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
    parser.add_argument('-scheduler_step', type=int, default=1000,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-normalize_loss', action='store_true',
                        help='Whether to normalize loss by taking attr_loss_weight into account')
    parser.add_argument('-use_relu', action='store_true',
                        help='Whether to include relu activation before using attributes to predict Y. '
                                'For end2end & bottleneck model')
    parser.add_argument('-use_sigmoid', action='store_true',
                        help='Whether to include sigmoid activation before using attributes to predict Y. '
                                'For end2end & bottleneck model')
    parser.add_argument('-connect_CY', action='store_true',
                        help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')
    args = parser.parse_args()
    args.three_class = (args.n_class_attr == 3)
    


    # Seeds
    np.random.seed(args[0].seed)
    torch.manual_seed(args[0].seed)
    
    arg = args[0]
    config_dict = vars(arg)
    print(config_dict)
    exit()
    experiment = args[0].exp
  
    wandb.init(project="img_processing", config=config_dict, name=f"{experiment}_{arg.seed}")
    wandb.config.update(arg)
    n_class_attr, n_attributes, expand_dim = config_dict
    
     # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(input_dim=n_attributes * n_class_attr, num_classes=N_CLASSES, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=n_attributes, num_classes=N_CLASSES, expand_dim=expand_dim)

    # {'dataset': 'cub',
    # 'exp': 'Independent_CtoY', 
    # 'seed': 1, 
    # 'log_dir': 'IndependentModel_WithVal___Seed1/outputs/',
    # 'batch_size': 64, 
    # 'epochs': 10, 
    # 'save_step': 1000, 
    # 'lr': 0.001,
    # 'weight_decay': 5e-05, 
    # 'pretrained': False,
    # 'freeze': False, 
    # 'use_aux': False,
    # 'use_attr': True,
    # 'attr_loss_weight': 1.0,
    # 'no_img': True, 
    # 'bottleneck': False,
    # 'weighted_loss': '', 
    # 'uncertain_labels': False, 
    # 'n_attributes': 112, 
    # 'expand_dim': 0,
    # 'n_class_attr': 2,
    # 'data_dir': 'src/datasets/CUB_processed/class_attr_data_10', 
    # 'image_dir': 'images', 
    # 'resampling': False,
    # 'end2end': False,
    # 'optimizer': 'sgd',
    # 'ckpt': '',
    # 'scheduler_step': 2,
    # 'normalize_loss': False,
    # 'use_relu': False, 
    # 'use_sigmoid': False, 
    # 'connect_CY': False,
    # 'three_class': False} 
   
   
   
   
   
    # Determine imbalance
    imbalance = None
    # Not needed for independent because  config_dict['no_img'] = True
    # if config_dict['use_attr'] and not config_dict['no_img'] and args.weighted_loss:
    #     train_data_path = os.path.join(BASE_DIR, config_dict['data_dir'], 'train.pkl')
    #     if args.weighted_loss == 'multiple':
    #         imbalance = find_class_imbalance(train_data_path, True)
    #     else:
    #         imbalance = find_class_imbalance(train_data_path, False)

    if os.path.exists(config_dict['log_dir']): # job restarted by cluster
        for f in os.listdir(config_dict['log_dir']):
            os.remove(os.path.join(config_dict['log_dir'], f))
    else:
        os.makedirs(config_dict['log_dir'])

    logger = Logger(os.path.join(config_dict['log_dir'], 'log.txt'))
    logger.write(str(config_dict) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss() # Loss function defined here
    
    # config_dict['no_img'] = True
    attr_criterion = None
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config_dict['lr'], momentum=0.9, weight_decay=config_dict['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config_dict['scheduler_step'], gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / config_dict['lr']) / math.log(LR_DECAY_SIZE)) * config_dict['scheduler_step']
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, config_dict['data_dir'], 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    if config_dict['ckpt']: #retraining
        train_loader = load_data([train_data_path, val_data_path], config_dict['use_attr'], config_dict['no_img'], config_dict['batch_size'], config_dict['uncertain_labels'], image_dir=config_dict['image_dir'], \
                                 n_class_attr=config_dict['n_class_attr'], resampling=config_dict['resampling'])
        val_loader = None
    else:
        train_loader = load_data([train_data_path], config_dict['use_attr'], config_dict['no_img'], config_dict['batch_size'], config_dict['uncertain_labels'], image_dir=config_dict['image_dir'], \
                                 n_class_attr=config_dict['n_class_attr'], resampling=config_dict['resampling'])
        val_loader = load_data([val_data_path], config_dict['use_attr'], config_dict['no_img'], config_dict['batch_size'], image_dir=config_dict['image_dir'], n_class_attr=config_dict['n_class_attr'])

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, config_dict['epochs']):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        #  config_dict['no_img'] = True for independent
        
        
        
       
        train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True)

        if not args.ckpt: # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
        
            with torch.no_grad():
                val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
                
        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(config_dict['log_dir'], 'best_model_%d.pth' % config_dict['seed']))
            #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\n'
                % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch)) 
        logger.flush()
        wandb.log({
        'epoch': epoch,
        'train_loss': train_loss_avg,
        'train_acc': train_acc_meter.avg,
        'val_loss': val_loss_avg,
        'val_acc': val_acc_meter.avg,
        'lr': scheduler.get_last_lr()[0]  # or optimizer.param_groups[0]['lr']
})

        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_lr())

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(config_dict['log_dir'], '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break
        
    wandb.finish()
    