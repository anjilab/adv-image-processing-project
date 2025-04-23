"""
Evaluate trained models on the official CUB test set
"""
import os
import sys
import torch
import joblib
import argparse
import numpy as np
from sklearn.metrics import f1_score
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tqdm

from dataset import load_data
from config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy
import torchvision
from pathlib import Path
import csv
from torch.utils.data import Dataset, DataLoader

K = [1, 3, 5] #top k class accuracies to compute

DEVICE = torch.device('cuda:3')
torch.cuda.set_device(DEVICE)

def denormalize(
    image: torch.Tensor, mean, std
) -> torch.Tensor:
    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    return image.mul_(std[:, None, None]).add_(mean[:, None, None])


def creating_adv(criterion, outputs, labels_var, model, inputs_var, epsilon):
    # inputs_var = inputs_var.to(DEVICE)
    loss = criterion(outputs, labels_var)
    model.zero_grad()
    loss.backward()
    
    grad_signs = torch.sign(inputs_var.grad)
    # pert_imgs = batch["pixel_values"] + 0.1 * grad_signs
    pert_imgs = inputs_var + epsilon * grad_signs
    pert_imgs = denormalize(pert_imgs, mean=[0.5, 0.5, 0.5],  std = [2, 2, 2])

    for img, name, perturbation in zip(pert_imgs, grad_signs):
        torchvision.utils.save_image(perturbation.double(), os.path.join(args.perturbed_imgs_dir, str(epsilon), name))
        torchvision.utils.save_image(img.double(), os.path.join(args.adv_imgs_dir, f"{epsilon}", name))
def eval(args):
    """
    Run inference using model (and model2 if bottleneck)
    Returns: (for notebook analysis)
    all_class_labels: flattened list of class labels for each image
    topk_class_outputs: array of top k class ids predicted for each image. Shape = size of test set * max(K)
    all_class_outputs: array of all logit outputs for class prediction, shape = N_TEST * N_CLASS
    all_attr_labels: flattened list of labels for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs: flatted list of attribute logits (after ReLU/ Sigmoid respectively) predicted for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs_sigmoid: flatted list of attribute logits predicted (after Sigmoid) for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    wrong_idx: image ids where the model got the wrong class prediction (to compare with other models)
    """
    if args.model_dir:
        model = torch.load(args.model_dir, weights_only=False).to(DEVICE)
        # print(next(model.parameters()).device, '======FOR MODEL PARAMETERS=========')
    else:
        model = None

    if not hasattr(model, 'use_relu'):
        if args.use_relu:
            model.use_relu = True
        else:
            model.use_relu = False
    if not hasattr(model, 'use_sigmoid'):
        if args.use_sigmoid:
            model.use_sigmoid = True
        else:
            model.use_sigmoid = False
    if not hasattr(model, 'cy_fc'):
        model.cy_fc = None
    model.eval()

    if args.model_dir2:
        if 'rf' in args.model_dir2:
            model2 = joblib.load(args.model_dir2).to(DEVICE)
        else:
            model2 = torch.load(args.model_dir2, weights_only=False).to(DEVICE)
        if not hasattr(model2, 'use_relu'):
            if args.use_relu:
                model2.use_relu = True
            else:
                model2.use_relu = False
        if not hasattr(model2, 'use_sigmoid'):
            if args.use_sigmoid:
                model2.use_sigmoid = True
            else:
                model2.use_sigmoid = False
        model2.eval()
    else:
        model2 = None
    if args.use_attr:
        attr_acc_meter = [AverageMeter()]
        if args.feature_group_results:  # compute acc for each feature individually in addition to the overall accuracy
            for _ in range(args.n_attributes):
                attr_acc_meter.append(AverageMeter())
    else:
        attr_acc_meter = None

    class_acc_meter = []
    for j in range(len(K)):
        class_acc_meter.append(AverageMeter())

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
    
     # Validate the existence of the directory to save the perturbed images
    if not os.path.exists(args.adv_imgs_dir):
        os.makedirs(args.adv_imgs_dir)
        
    test_loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr, augmentation_type=args.augmentation_type)

    epsilons = [0.01, 0.03, 0.1, 0.5]
    criterion = torch.nn.CrossEntropyLoss()
        
    for epsilon in epsilons:
        # Ensure directories exist
        Path(args.perturbed_imgs_dir, str(epsilon)).mkdir(parents=True, exist_ok=True)
        Path(args.adv_imgs_dir, str(epsilon)).mkdir(parents=True, exist_ok=True)

        # Open label CSV files if not already open
        perturbed_labels_file = open(os.path.join(args.perturbed_imgs_dir, str(epsilon), "labels.csv"), mode="a", newline='')
        adv_labels_file = open(os.path.join(args.adv_imgs_dir, str(epsilon), "labels.csv"), mode="a", newline='')
        
        perturbed_writer = csv.writer(perturbed_labels_file)
        adv_writer = csv.writer(adv_labels_file)
        
        all_outputs, all_targets = [], []
        all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, all_attr_outputs2 = [], [], [], []
        all_class_labels, all_class_outputs, all_class_logits = [], [], []
        topk_class_labels, topk_class_outputs = [], []
        # Validate the existence of the directory to save the perturbed images
         
        epsilon_dir = os.path.join(args.adv_imgs_dir, f"{epsilon}")
        epsilon_dir_perturbed= os.path.join(args.perturbed_imgs_dir, f"{epsilon}")
        
        if not os.path.exists(epsilon_dir_perturbed):
            os.makedirs(epsilon_dir_perturbed, exist_ok=True)  # Create the directory if it doesn't exist
            
        if not os.path.exists(epsilon_dir):
            os.makedirs(epsilon_dir, exist_ok=True)  # Create the directory if it doesn't exist
        count = 0
        for data_idx, data in enumerate(test_loader):
            if args.use_attr:
                print('Using attribute>>>>>>>>>>>.\n ')
                if args.no_img:  # A -> Y
                    print('args.no_img \n \n  ', args.no_img)
                    inputs, labels = data
                    if isinstance(inputs, list):
                        inputs = torch.stack(inputs).t().float()
                    inputs = inputs.float()
                    # inputs = torch.flatten(inputs, start_dim=1).float()
                else:
                    print('args.no_img', args.no_img)
                    inputs, labels, attr_labels = data
                    attr_labels = torch.stack(attr_labels).t()  # N x 312
            else:  # simple finetune
                inputs, labels = data
                
            inputs_var = inputs.to(DEVICE)
            labels_var = labels.to(DEVICE)
            
            inputs_var.requires_grad = True
            
            if args.attribute_group:
                print('Attribute group')
                outputs = []
                f = open(args.attribute_group, 'r')
                for line in f:
                    attr_model = torch.load(line.strip())
                    outputs.extend(attr_model(inputs_var))
            else:
                outputs = model(inputs_var)
            
            if args.use_attr: # FOr independent model, it enters here.
                print('independent -> true')
                if args.no_img:  # A -> Y
                    print('independent ->  false')
                    class_outputs = outputs
                else:
                    if args.bottleneck:
                        print('Is bottleneck: ', args.bottleneck)
                        print('Use relu ?: ', args.use_relu)
                        print('Use sigmoid ?: ', args.use_sigmoid)
                        
                        
                        if args.use_relu:
                            attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
                            attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                        elif args.use_sigmoid:
                            attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
                            attr_outputs_sigmoid = attr_outputs
                        else:
                            attr_outputs = outputs
                            attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                        if model2:
                            print('Independent')
                            stage2_inputs = torch.cat(attr_outputs, dim=1)
                            class_outputs = model2(stage2_inputs)
                        
                
                            
                        else:  # for debugging bottleneck performance without running stage 2
                            class_outputs = torch.zeros([inputs.size(0), N_CLASSES],
                                                        dtype=torch.float64).cuda()  # ignore this
                    else:  # cotraining, end2end
                        if args.use_relu:
                            attr_outputs = [torch.nn.ReLU()(o) for o in outputs[1:]]
                            attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                        elif args.use_sigmoid:
                            attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                            attr_outputs_sigmoid = attr_outputs
                        else:
                            attr_outputs = outputs[1:]
                            attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]

                        class_outputs = outputs[0]
                    for i in range(args.n_attributes):
                        acc = binary_accuracy(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
                        acc = acc.data.cpu().numpy()
                        # acc = accuracy(attr_outputs_sigmoid[i], attr_labels[:, i], topk=(1,))
                        attr_acc_meter[0].update(acc, inputs.size(0))
                        if args.feature_group_results:  # keep track of accuracy of individual attributes
                            attr_acc_meter[i + 1].update(acc, inputs.size(0))

                    attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1)
                    attr_outputs_sigmoid = torch.cat([o for o in attr_outputs_sigmoid], dim=1)
                    all_attr_outputs.extend(list(attr_outputs.flatten().data.cpu().numpy()))
                    all_attr_outputs_sigmoid.extend(list(attr_outputs_sigmoid.flatten().data.cpu().numpy()))
                    all_attr_labels.extend(list(attr_labels.flatten().data.cpu().numpy()))
                    
                    # print(inputs_var.shape, class_outputs.shape, labels_var.shape, attr_outputs.shape, labels_var)
                    loss = criterion(class_outputs, labels_var)
                    model.zero_grad()
                    loss.backward()
                    grad_signs = torch.sign(inputs_var.grad)
                    pert_imgs = inputs_var + epsilon * grad_signs
                    pert_imgs = denormalize(pert_imgs, mean=[0.5, 0.5, 0.5],  std = [2, 2, 2])
                    for index, (img, perturbation) in enumerate(zip(pert_imgs, grad_signs)):
                        try:
                            torchvision.utils.save_image(perturbation.double(), os.path.join(args.perturbed_imgs_dir, str(epsilon), f"{count + data_idx}.png"))                        
                            torchvision.utils.save_image(img.double(), os.path.join(args.adv_imgs_dir, f"{epsilon}", f"{count + data_idx}.png"))
                            perturbed_writer.writerow([f"{count + data_idx}.png", labels_var[index].item()])
                            if args.use_attr:
                                adv_writer.writerow([f"{count + data_idx}.png", labels_var[index].item(), attr_labels[index].tolist()])
                            else:
                                adv_writer.writerow([f"{count + data_idx}.png", labels_var[index].item()])
                        except Exception as e:
                            print('error in saving')
                        count = count + 1 
            else:
                class_outputs = outputs[0]
                
                loss = criterion(class_outputs, labels_var)
                model.zero_grad()
                loss.backward()
                grad_signs = torch.sign(inputs_var.grad)
                # pert_imgs = batch["pixel_values"] + 0.1 * grad_signs
                pert_imgs = inputs_var + epsilon * grad_signs
                pert_imgs = denormalize(pert_imgs, mean=[0.5, 0.5, 0.5],  std = [2, 2, 2])
                for index, (img, perturbation) in enumerate(zip(pert_imgs, grad_signs)):
                    # print(index, img.shape, perturbation.shape, )
                    torchvision.utils.save_image(perturbation.double(), os.path.join(args.perturbed_imgs_dir, str(epsilon), f"{count + data_idx}.png"))                        
                    torchvision.utils.save_image(img.double(), os.path.join(args.adv_imgs_dir, f"{epsilon}", f"{count + data_idx}.png"))
                    perturbed_writer.writerow([f"{count + data_idx}.png", labels_var[index].item()])
                    if args.use_attr:
                        adv_writer.writerow([f"{count + data_idx}.png", labels_var[index].item(), attr_labels[index].item()])
                    else:
                        adv_writer.writerow([f"{count + data_idx}.png", labels_var[index].item()])
                        
                    count = count + 1
                
                
if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, nargs='+', help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')    
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-augmentation_type', default='standard', help='Augmentation')
    parser.add_argument('-adv_imgs_dir', default='src/datasets/CUB_processed_adversarial', help='Adv image directory')
    parser.add_argument('-perturbed_imgs_dir', default='src/datasets/CUB_processed_adversarial_perturb', help='Perturbed image')
    
    
    
    args = parser.parse_args()
    args.batch_size = 16

    y_results, c_results = [], []
    for i, model_dir in enumerate(args.model_dirs):
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None
        result = eval(args)
