This repository contains the code for the class project of  concept bottleneck model analysis via adversarial attacks and image processing techniques. 

To reproduce the results shown in presentation. 

1. You need to run ```bash scripts_joint_model.sh``` for the training process and results without adversarial attacks.
2. To create the adversarial dataset, use ```bash adv_scripts.sh```.
3. To generate the result for different analysis and original model, use ```bash inference_report.sh```.


### Project Steps

1. `uv venv --python 3.10 --seed `
2. `source .venv/bin/activate`
3. Download datasets from `https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683` 
4. Inside src, `mkdir datasets/CUB_processed` and then `tar -xvzf CUB_processed.tar.gz -C /media/drive2/anjilabudathoki/codes/adv-image-processing-project/src/datasets/CUB_processed/`
5. `uv pip install -r requirements.txt` 
6. `uv pip install torchsummary`
<!-- 8. `source .venv/bin/activate && uv pip install torch==2.4.0 torchvision==0.19.0` -->
