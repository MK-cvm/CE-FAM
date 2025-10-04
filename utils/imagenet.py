import os
import csv
import torch
import json
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

class EvalDataset(ImageFolder):
    def __init__(self, label_root, name_list, transform, metafile, processor):
        super(EvalDataset, self).__init__(label_root)
        self.label_list = []
        self.image_list = []
        self.transform = transform
        self.processor = processor
        with open(name_list, 'r') as f:
            names = f.read().splitlines()
            for name in names:
                image_file = name.split(' ')[0]
                gt_file = name.split(' ')[1]
                self.label_list.append(os.path.join(label_root, 'validation-segmentation', gt_file))
                self.image_list.append(os.path.join(label_root, 'validation', image_file))
        with open(metafile, mode='r') as f:
            self.wnid_to_id = json.load(f)

    def __getitem__(self, item):
        image_file = self.image_list[item]
        dirname = os.path.dirname(image_file)
        filename = os.path.splitext(os.path.basename(image_file))[0]
        wnid = dirname.split('/')[-1]
        gt_id = self.wnid_to_id[wnid][0]
        image = Image.open(image_file)
        gt = Image.open(self.label_list[item])
        gt = np.array(gt)
        gt = gt[:, :, 1] * 256 + gt[:, :, 0]
        gt_mask = self.get_mask_of_class(gt)
        image = self.transform(image)
        gt_mask = Image.fromarray(gt_mask)
        gt_mask = self.transform(gt_mask).to(torch.bool)
        return image, gt_mask, gt_id, filename
    
    def get_mask_of_class(self, mask):
        ids = np.unique(mask)
        mask_v = (mask == ids[-1])
        return mask_v

    def __len__(self):
        return len(self.label_list)

def get_imagenet_concept(concept_dim, tokenizer, vl_model, vl_model_name, device):
    concept_label_path = os.path.join('concepts','ImageNetS', 'imagenet_classes.txt')
    concept_list = []
    concept_type_dict = {}
    with open(os.path.join(concept_label_path), newline='') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            concept = row[0]
            concept_type_dict[concept] = 'object'
            concept_list.append(concept)
    num_concept = len(concept_list)
    concept_vectors = torch.zeros((num_concept, concept_dim), dtype=torch.float).to(device)
    for idx, concept in enumerate(concept_list):
        with torch.no_grad():
            concept_temp = "a photo of a %s" % (concept)
            if vl_model_name == 'clip':
                concept_id = tokenizer(concept_temp, return_tensors="pt").to(device)
            elif vl_model_name == 'siglip':
                concept_id = tokenizer(concept_temp, padding="max_length", return_tensors="pt").to(device)
            text_feat = vl_model.get_text_features(**concept_id)
            concept_vector = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            concept_vector = concept_vector.to(device)
            concept_vectors[idx] = concept_vector
    return num_concept, concept_vectors, concept_list, concept_type_dict