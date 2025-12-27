# CE-FAM: Concept-based Explanation via Fusion of Activation Maps
This is the official repository for our paper ["CE-FAM: Concept-Based Explanation via Fusion of Activation Maps"](https://arxiv.org/abs/2509.23849), published at ICCV 2025.

# 1. Dataset Preparation
### 1. Broden
- Please move `concepts/broden/dlbroden.sh` to the root directory and then run it.
### 2. ImageNet-S
- Please prepare the dataset according to the [official code](https://github.com/LUSSeg/ImageNet-S).
# 2. Concept Training
Please refer to `scripts/command_train.sh` for an example command.

# 3. Concept Evaluation
Please refer to `scripts/command_evaluate.sh` for an example command.

# (Optional) Concept Matching in Existing Methods
- Please use the official code ([CLIP-Dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect), [WWW](https://github.com/ailab-kyunghee/WWW)) and select an appropriate image classifier, dataset, concept label, and target layer.
- To unify the concept labels, please use `concepts/broden/broden_labels_clean.txt` for Broden and `concepts/ImageNetS/imagenet_classes.txt` for ImageNet.
- Please use the following models for the image classifiers.

    | ResNet50          |  ResNet18        | ViT-B/16        |
    | ------------------ | ----------- | ---------- |
    | Pytorch Model <br> Weights.IMAGENET1K_V2 | Pytorch Model <br> Weights.IMAGENET1K_V1 | Pytorch Model <br> Weights.IMAGENET1K_V1 |    

- Please rename the obtained concept-matching files as appropriate and store them in the directory below.
```
CEFAM
├── concepts
      ├── CLIP-dissect
      |     ├── resnet50
      |           ├── broden/descriptions.csv
      |           └── imagenet/descriptions.csv     
      ├── WWW    
            ├── resnet50
                  ├── www_1k_tem_adp_95_fc.pkl
                  ├── www_1k_tem_adp_95_l4.pkl
                  ├── www_1k_tem_adp_95_l3.pkl
                  ├── www_1k_tem_adp_95_l2.pkl
                  ├── www_1k_tem_adp_95_l1.pkl
                  ├── www_broadenk_tem_adp_95_l4.pkl
                  ├── www_broadenk_tem_adp_95_l3.pkl
                  ├── www_broadenk_tem_adp_95_l2.pkl
                  └── www_broadenk_tem_adp_95_l1.pkl
``` 
