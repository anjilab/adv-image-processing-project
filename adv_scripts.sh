# Using joint model to create adversarial and testing on joint model only

# python3 src/adv-bird-data-create.py -model_dirs Joint-models-log/Joint0.001Model_Seed1/outputs/best_model_1.pth Joint-models-log/Joint0.001Model_Seed2/outputs/best_model_2.pth Joint-models-log/Joint0.001Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/datasets/CUB_processed/class_attr_data_10 -bottleneck -use_sigmoid -log_dir Joint-models-log/Joint0.001Model/outputs -adv_imgs_dir src/datasets/CUB_processed_adversarial_jointmdl -perturbed_imgs_dir src/datasets/CUB_processed_adversarial_perturb_jointmdl
# python3 src/adv-bird-data-create.py  -model_dirs Joint-models-log/Joint0.001Model_Seed1/outputs/best_model_1.pth Joint-models-log/Joint0.001Model_Seed2/outputs/best_model_2.pth Joint-models-log/Joint0.001Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/datasets/CUB_processed/class_attr_data_10 -log_dir Joint-models-log/Joint-models-log/Joint0.1Model/outputs -adv_imgs_dir src/datasets/CUB_processed_adversarial_jointmdl -perturbed_imgs_dir src/datasets/CUB_processed_adversarial_perturb_jointmdl

# Successfully created image for use_attribute=true in case of joint models
# python3 src/adv-bird-data-create.py  -model_dirs Joint-models-log/Joint0.001Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/datasets/CUB_processed/class_attr_data_10 -log_dir Joint-models-log/Joint-models-log/Joint0.001Model/outputs/adv-img -adv_imgs_dir src/datasets/CUB_processed_adversarial_jointmdl -perturbed_imgs_dir src/datasets/CUB_processed_adversarial_perturb_jointmdl


#RUN THISN AFTER FINISHING ABOVE, WANT TO SAVE ATTRIBUTE GT TOO.
python3 src/adv-bird-data-create.py  -model_dirs Joint-models-log/Joint0.001Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir src/datasets/CUB_processed/class_attr_data_10 -log_dir Joint-models-log/Joint-models-log/Joint0.001Model/outputs/adv-img -adv_imgs_dir src/datasets/CUB_processed_adversarial_jointmdl_attr_save -perturbed_imgs_dir src/datasets/CUB_processed_adversarial_perturb_jointmdl_attr_save

