import os
import csv
import pickle
import torch
import cv2
import glob
from tqdm import tqdm
import numpy as np

def get_broden_concept(concept_label_path, concept_dim, tokenizer, vl_model, vl_model_name, device):
    concept_list, concept_type_dict = get_concepts(concept_label_path)
    num_concept = len(concept_list)
    concept_vectors = torch.zeros((num_concept, concept_dim), dtype=torch.float).to(device)
    for idx, (concept, concept_type) in enumerate(concept_list):
        with torch.no_grad():
            if concept_type == 'color':
                concept_temp = "a photo of a %s object" % (concept)
            elif concept_type == 'material':
                concept_temp = "a photo of an object made of %s" % (concept)
            elif concept_type == 'object':
                concept_temp = "a photo of a %s" % (concept)
            elif concept_type == 'part':
                concept_temp = "a photo of a %s" % (concept)
            elif concept_type == 'scene':
                concept_temp = "a photo of an object on %s" % (concept)
            elif concept_type == 'texture':
                concept_temp = "a photo of a %s object" % (concept)
            if vl_model_name == 'clip':
                concept_id = tokenizer(concept_temp, return_tensors="pt").to(device)
            elif vl_model_name == 'siglip':
                concept_id = tokenizer(concept_temp, padding="max_length", return_tensors="pt").to(device)
            text_feat = vl_model.get_text_features(**concept_id)
            concept_vector = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            concept_vector = concept_vector.to(device)
            concept_vectors[idx] = concept_vector
    return num_concept, concept_vectors, concept_list, concept_type_dict

def get_concepts(base_path):
    concept_list = []
    concept_type_dict = {}
    with open(os.path.join(base_path, 'label.csv'), newline='') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0: continue
            concept_type = row[2].split('(')[0]
            concept = row[1]
            if concept_type == 'scene':
                temp = list(concept)
                temp[-2:] = ''
                concept = ''.join(temp)
            elif concept_type == 'color':
                temp = list(concept)
                temp[-2:] = ''
                concept = ''.join(temp)
            concept = concept.replace('_', ' ')
            concept_type_dict[concept] = concept_type
            concept_list.append((concept, concept_type))
    return concept_list, concept_type_dict

def decodeClassMask(im):
    return im[:,:,1] * 256 + im[:,:,0]

def make_label_list(base_dir):
    data_types = ['ade20k', 'dtd', 'opensurfaces', 'pascal']
    concept_list, _ = get_concepts(base_dir)
    label_list = []
    for data_type in data_types:
        print('processing %s data'%(data_type))
        image_files = sorted(glob.glob(os.path.join(base_dir, 'images', data_type, "*.png")))
        for image_file in tqdm(image_files):
            filename = os.path.splitext(os.path.basename(image_file))[0].split('_')
            mask_bgr = cv2.imread(image_file)
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB).astype(np.uint16)
            mask_class = decodeClassMask(mask_rgb)
            concept_ids = np.unique(mask_class)
            if data_type in ['ade20k']:
                base_image_file = "%s_%s_%s.jpg"%(filename[0], filename[1],filename[2])
            elif data_type in ['dtd', 'pascal']:
                base_image_file = "%s_%s.jpg"%(filename[0], filename[1])
            elif data_type in ['opensurfaces']:
                base_image_file = "%s.jpg"%(filename[0])
            label_dict = {}
            label_dict['path'] = os.path.join(os.path.dirname(image_file),base_image_file)
            label_dict['label'] = {}
            for concept_id in concept_ids:
                if concept_id == 0: continue
                concept_name, concept_type = concept_list[concept_id-1]
                mask = (mask_class == concept_id)
                label_dict['label'][concept_id] = (mask, concept_name, concept_type)
            label_list.append(label_dict)
    with open(os.path.join(base_dir, 'broden_concept_labels.pkl'), mode='wb') as fo:
        pickle.dump(label_list, fo)
    
    return label_list