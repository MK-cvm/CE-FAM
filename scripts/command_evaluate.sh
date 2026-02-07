#/bin/bash

# Broden Top-4
# python3 evaluate.py --eval_mode region --concept_dataset broden --probe_layer all --vl_model_name clip --ckpt_path PATH_TO_YOUR_MODEL
python3 evaluate.py --eval_mode region --concept_dataset broden --probe_layer all --probe_model_name resnet50 --vl_model_name clip --ckpt_path ./ckpt/4layer_similarityloss_broden_resnet50_clip/best_converter_ckpt.pth
python3 evaluate.py --eval_mode region --concept_dataset broden --probe_layer all --probe_model_name resnet18 --vl_model_name clip --ckpt_path ./ckpt/4layer_similarityloss_broden_resnet18_clip/best_converter_ckpt.pth
python3 evaluate.py --eval_mode region --concept_dataset broden --probe_layer encoder --probe_model_name vit_b_16 --vl_model_name clip --single_layer --layer_cand_num 1 --ckpt_path ./ckpt/1layer_similarityloss_broden_vitb16_clip/best_converter_ckpt.pth

# Broden Top-1
# python3 evaluate.py --eval_mode region --concept_dataset broden --probe_layer all --layer_cand_num 1 --vl_model_name clip --ckpt_path PATH_TO_YOUR_MODEL

# ImageNet
# python3 evaluate.py --eval_mode region --concept_dataset imagenet --probe_layer layer4 --layer_cand_num 1 --vl_model_name clip --single_layer --ckpt_path PATH_TO_YOUR_MODEL
python3 evaluate.py --eval_mode region --concept_dataset imagenet --probe_layer layer4 --probe_model_name resnet50 --layer_cand_num 1 --vl_model_name clip --single_layer --ckpt_path ./ckpt/1layer_similarityloss_imagenet_resnet50_clip/best_converter_ckpt.pth
python3 evaluate.py --eval_mode region --concept_dataset imagenet --probe_layer layer4 --probe_model_name resnet18 --layer_cand_num 1 --vl_model_name clip --single_layer --ckpt_path ./ckpt/1layer_similarityloss_imagenet_resnet18_clip/best_converter_ckpt.pth
python3 evaluate.py --eval_mode region --concept_dataset imagenet --probe_layer encoder --probe_model_name vit_b_16 --layer_cand_num 1 --vl_model_name clip --single_layer --ckpt_path ./ckpt/1layer_similarityloss_imagenet_vitb16_clip/best_converter_ckpt.pth

# Contribution
# python3 evaluate.py --eval_mode contribution --concept_dataset broden --probe_layer all --vl_model_name clip --ckpt_path PATH_TO_YOUR_MODEL
