#/bin/bash

python3 train.py --train_batch 256 --model_type clip --concept_name broden --probe_model resnet50 --score_loss --init_lr 0.1 --warmup_epoch 0