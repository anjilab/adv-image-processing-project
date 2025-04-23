# Steps
1. I copied all the folders from CUB to my folders. 
2. Then create new script file as run_command_2025.sh
3. I needed to copy all the files associated with the CUB.
4. Next when i ran, i was asked for CUB_processed dataset.
5. Then i was asked for CUB_200_211/images/train
6. Using debugger, first we go to experiments.py that will first train concept models i.e mapping input to concept. Uses, inception v3, all layers are trainable ~ 24M parameter. 
7. While  we train the model, using CUB_processed. Pay attention to lot of things.
    a. First they check imabalance in dataset. What is the role of weighted loss ? This is further used in finding imbalance with multiple attribute True or False condition.
    b. They define model loss.
    c. They had args.use_attr and not args.no_img and defined attribute loss  based on args.weighted_loss, What is the purpose here ? Once attribute loss are stored. note: for args.weighted loss set to multiple they used bcewithlogitsloss otherwise normal CEentropyloss, what is the need ? 
    d. Then they moved to model optimizer.
    e. Moved to scheduler. 
    f. Then setting path of train and validation path and then based on args.ckpt, we load the data if this is true and then train loader gonna have both train and val path but if there is no ckpt we have different train loader and val loader. 
    g. Here then to load data, load_data from dataset.py is used where first some transformation is done. Once the preprocessing transformation is done, we will need dataset, this dataaset is managed in CUBDataset. 
    h. While running each image in epoch, it will go for __getitem__ in CUBdataset
    i. Now it requires CUB_200_2011 Dataset, why i believe is it for attribute. 
    k. During training process, i got error at CUB_200_2011/images/train/110.Geococcyx/Geococcyx_0028_104751.jpg, bUT THIS DATA IS PRESENT. 

8. While running bash run_command.sh, to create ExtractConcepts we need to uncomment device in generate_new_data.py


# Some concepts of code and paper
Number of attributes and concepts are same.
Using Inception3 => this is base pretrained model for feature extraction i guess or for everything check ? 
For this try with other models and see what happens ? 
We are training on CUB_processed dataset, class imbalance has also been dealt with.
Find out what is the purpose of imbalance and different loss function. 

Optimizer =  SGD


# Some confusion i had while understanding code
1. i pushed to cuda:2 but still the cpu usage was skyrocketing why ? 

# tasks
Joint model use adversarail 
COMPOSITE LOSS 

adversarial image create by using independent models

3 different adv image create.


TASKS: 
1. Joint Model train train, val accuracy => train and validation might be same check. =>DONE checked issue fixed, choosen joint model with 0.001 as now adversarial image creation
2. 3 different set of adversarial example
 - Concept = 0
 - Task = 0 
 - concept + task loss  => main  [ADV IMAGE CREATION ON THE PROCESS]
3. Evaluation of joint model on the adversarial dataset. 
4. Check joint model y error and c error. 

Result interpretation:
Precision: For class 1: if it is 0.59 means of all the times model predicted 1, how many times is actually 1. Avoid false positive
Recall: For class 1: if it sis 0.50, of all the actual ones, how many times did the model actually predicted 1. Helps in minority class. 

My models precision:
for class 0 = 0.88, class 1 = 0.59. This means model is leaning towards class 0. 
F1 score = 0.54, means that model is missing class 1 cases and not super confident.  


