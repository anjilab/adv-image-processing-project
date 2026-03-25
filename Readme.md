# An Exploratory Analysis of the Impacts of Image Processing Techniques and Adversarial Method on Concept Bottleneck Models

## Introduction

This repository contains the implementation and analysis of **Concept Bottleneck Models (CBMs)** under adversarial attacks and image processing techniques. The project explores the robustness and interpretability of CBMs when subjected to adversarial perturbations, using the CUB-200-2011 bird species classification dataset.


### What are Concept Bottleneck Models?

Concept Bottleneck Models (CBMs) aim to enhance interpretability in machine learning by introducing an intermediate layer of human-understandable concepts between inputs and predictions i.e  decompose the prediction process into two stages:
1. **X → C**: Map input images to human-understandable concept/attribute predictions (e.g., "has red wings", "has long beak")
2. **C → Y**: Map predicted concepts to final class labels (e.g., bird species)

This architecture allows for better interpretability and intervention capabilities compared to traditional end-to-end models.


### Project Objectives

- Train and evaluate joint concept bottleneck models on bird species classification
- Generate adversarial examples using various attack techniques
- Analyze model robustness against adversarial perturbations
- Evaluate the impact of adversarial attacks on both concept prediction and final classification accuracy


### Dataset download

1. Download datasets for CUB_processed: `https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683` and for CUB_200_2011: `https://worksheets.codalab.org/bundles/0xd013a7ba2e88481bbc07e787f73109f5`, and pretrained model `https://worksheets.codalab.org/bundles/0x3ee72bbd60144902aa8e90f2f6f462f3`
2. Inside src, `mkdir datasets/CUB_processed` and then `tar -xvzf CUB_processed.tar.gz -C /media/drive2/anjilabudathoki/codes/adv-image-processing-project/src/datasets/CUB_processed/` to get CUB_processed
3. Run `tar -xvzf CUB_200_2011.tgz -C /media/drive2/anjilabudathoki/codes/adv-image-processing-project/src/datasets/` to get CUB_200_2011
4. Run `tar -xvzf pretrained.tar.gz -C src/pretrained-model` for the model

## Setup Instructions

### 1. Environment Setup

1. `uv venv --python 3.10 --seed `
2. `source .venv/bin/activate`
3. `uv pip install -r requirements.txt` 
4. `uv pip install torchsummary`

**Note**: Based on your CUDA version, PyTorch installation may need to be adjusted. The experiments were conducted on CUDA Version 12.6 using a single NVIDIA RTX 4090 GPU from a 6-GPU setup.

<!-- 8. `source .venv/bin/activate && uv pip install torch==2.4.0 torchvision==0.19.0` -->


<!-- 
To reproduce the results shown in report. 

1. You need to run ```bash scripts_joint_model.sh``` for the training process and results without adversarial attacks.
2. To create the adversarial dataset, use ```bash adv_scripts.sh```. Before running this script, you should run above 1. otherwise it won't work. 
3. To generate the result for different analysis and original model, use ```bash inference_report.sh```. We need to make sure the outputs are there to get any results, see the scripts file itself.  -->


## Reproducing Results

1. Run `bash scripts_joint_model.sh` to execute the training process and generate results without adversarial attacks
2. Run `bash adv_scripts.sh` to create the adversarial dataset (requires completing step 1 first)
3. Run `bash inference_report.sh` to generate analysis results for different configurations and the original model (ensure model outputs exist before running)


## Results and Visualization

To see screenshots and visualizations of the results, refer to the project presentation: [View Slides](concept_bottleneck_models.pdf)


