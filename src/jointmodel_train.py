
from template_model import MLP, inception_v3, End2EndModel
import torch

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
from config import BASE_DIR, N_CLASSES,  MIN_LR, LR_DECAY_SIZE
import wandb

from torchsummary import summary

DEVICE = torch.device('cuda:2')
torch.cuda.set_device(DEVICE)

def get_params_info(model1, freeze, n_attributes, expand_dim ):
    for param in model1.parameters():
        param.requires_grad = not freeze  # If freeze=True, this will stop training on base layers
    num_trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"Number of trainable parameters of model1 inception: {num_trainable_params}") # 24658880 
    model2 = MLP(input_dim=n_attributes, num_classes=N_CLASSES, expand_dim=expand_dim)
    num_trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"Number of trainable parameters of model2 mlp: {num_trainable_params}") # 22600


def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
           # Enters Joint model
            inputs, labels, attr_labels = data
            if args.n_attributes > 1:
                # Enters Joint model
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()#.float() #N x 312 # GT LABELS SHAPE [bs, n_attribute] = [64, 112]
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            
            # attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels.float().requires_grad_()
            attr_labels_var = attr_labels_var.to(DEVICE)
            # attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        # inputs_var = torch.autograd.Variable(inputs)
        # inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        # labels_var = torch.autograd.Variable(labels)
        # labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        inputs_var = inputs.to(DEVICE)
        labels_var = labels.to(DEVICE)
       

        if is_training and args.use_aux:
            # Joint model during training
            outputs, aux_outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                # Joint model enters here
                # print('heeeeeeere args.bottleneck', not args.bottleneck)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            # print(attr_criterion is not None and args.attr_loss_weight > 0, '--------sp many loss append')
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                # Joint enters here tooo
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[i+out_start].squeeze().to(DEVICE).float(), attr_labels_var[:, i]) \
                                                            + 0.4 * attr_criterion[i](aux_outputs[i+out_start].squeeze().to(DEVICE).float(), attr_labels_var[:, i])))
        else: #testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))
    
        if args.bottleneck: #attribute accuracy
            # Joint do not enter here
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            # print('inside accuracy calculation else side')
            acc = accuracy(outputs[0], labels, topk=(1,)) # only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses)/ args.n_attributes
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                # Joint should ender here
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else: #finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, acc_meter


# Joint Model with parameter = 24681480
# Processing takes input image -> 2048 dim vector -> 112 FC
def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim,
                 use_relu, use_sigmoid):
    # Taking pretrained model [X to Y, directly 200 classes prediction]
    model1 = inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                          n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                          three_class=(n_class_attr == 3)) 
    
    model2 = MLP(input_dim=n_attributes, num_classes=N_CLASSES, expand_dim=expand_dim)
    # get_params_info(model1, freeze, n_attributes, expand_dim)
    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)

def train_X_to_C_to_y(args):
    # Joint model [End to end]
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    
    train(model, args)
    
    
def train(model, args):
    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)
    if os.path.exists(args.log_dir): # job restarted by cluster
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    model = model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img: # True and not False 
        # Joint model enters here
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert(imbalance is not None)
            for ratio in imbalance:
                # prediction = [bs, 112], target = [bs, 112], weight = [112] here this means imbalance
                # loss will be calculated as loss[i][j] = weight[j] * BCEWithLogits(pred[i][j], target[i][j])
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda())) # 112 separate losses with their own imbalance ratio weight scale


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    # if args.ckpt: #retraining
    #     train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
    #                              n_class_attr=args.n_class_attr, resampling=args.resampling)
    #     val_loader = None
    # else:
    train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, n_class_attr=args.n_class_attr, resampling=args.resampling, augmentation_type=args.augmentation_type)
    val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0
    
    print(len(train_loader.dataset), len(val_loader.dataset))

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_loss_meter, train_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True) 
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        with torch.no_grad():
            val_loss_meter, val_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion, args, is_training=False)


        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))
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
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break
        
    wandb.finish()
