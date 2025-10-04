#/bin/bash

# Broden Top-4
# python3 evaluate.py --eval_mode region --concept_dataset broden --probe_layer all --vl_model_name clip --ckpt_path PATH_TO_YOUR_MODEL

# Broden Top-1
# python3 evaluate.py --eval_mode region --concept_dataset broden --probe_layer all --layer_cand_num 1 --vl_model_name clip --ckpt_path PATH_TO_YOUR_MODEL

# ImageNet
# python3 evaluate.py --eval_mode region --concept_dataset imagenet --probe_layer layer4 --layer_cand_num 1 --vl_model_name clip --single_layer --ckpt_path PATH_TO_YOUR_MODEL

# Contribution
# python3 evaluate.py --eval_mode contribution --concept_dataset broden --probe_layer all --vl_model_name clip --ckpt_path PATH_TO_YOUR_MODEL
