import os
import pickle
import cv2
import copy
import numpy as np

def get_www_dict(concept_dataset, probe_model_name, probe_layer, concept_type_dict, object_mode=False):
    concept_layer_dict = {}
    www_files = []
    if concept_dataset == 'imagenet':
        if probe_layer in ['fc', 'all']:
            www_files.append(('www_1k_tem_adp_95_fc.pkl', 'fc'))
        if probe_layer in ['layer4', 'all']:
            www_files.append(('www_1k_tem_adp_95_l4.pkl', 'layer4'))
        if probe_layer in ['layer3', 'all']:
            www_files.append(('www_1k_tem_adp_95_l3.pkl', 'layer3'))
        if probe_layer in ['layer2', 'all']:
            www_files.append(('www_1k_tem_adp_95_l2.pkl', 'layer2'))
        if probe_layer in ['layer1', 'all']:
            www_files.append(('www_1k_tem_adp_95_l1.pkl', 'layer1'))
        if probe_layer in ['encoder']:
            www_files.append(('www_1k_tem_adp_95_vit.pkl', 'encoder'))
    elif concept_dataset == 'broden':
        if probe_layer in ['layer4', 'all']:
            www_files.append(('www_broadenk_tem_adp_95_l4.pkl', 'layer4'))
        if probe_layer in ['layer3', 'all']:
            www_files.append(('www_broadenk_tem_adp_95_l3.pkl', 'layer3'))
        if probe_layer in ['layer2', 'all']:
            www_files.append(('www_broadenk_tem_adp_95_l2.pkl', 'layer2'))
        if probe_layer in ['layer1', 'all']:
            www_files.append(('www_broadenk_tem_adp_95_l1.pkl', 'layer1'))
        if probe_layer in ['encoder']:
            www_files.append(('www_broadenk_tem_adp_95_vit.pkl', 'encoder'))

    for www_file, layer in www_files:
        with open(os.path.join('concepts', 'WWW', probe_model_name, www_file), 'rb') as f:
            concept, concept_weight = pickle.load(f)
            for idx, (layer_concepts, layer_concept_weights) in enumerate(zip(concept, concept_weight)):
                for layer_concept, layer_concept_weight in zip(layer_concepts, layer_concept_weights):
                    if layer_concept.endswith("-c"): layer_concept = layer_concept[:-2]
                    assert layer_concept in concept_type_dict.keys()
                    if layer_concept not in concept_layer_dict.keys():
                        concept_layer_dict[layer_concept] = [(layer_concept_weight, layer, idx)]
                    else:
                        concept_layer_dict[layer_concept].append((layer_concept_weight, layer, idx))
    for key, value in concept_layer_dict.items():
        value.sort(key=lambda x: x[0], reverse=True)
    return concept_layer_dict, concept

def region_www(feature_layers, mask, www_dict, concept_name, layer_cand_num):
    act_maps = []
    layers = []
    map_descriptions = []
    for idx, (similarity, layer, index) in enumerate(www_dict[concept_name]):
        if layer == 'layer4': act_map = feature_layers[3][0, index]
        elif layer == 'layer3': act_map = feature_layers[2][0, index]
        elif layer == 'layer2': act_map = feature_layers[1][0, index]
        elif layer == 'layer1': act_map = feature_layers[0][0, index]
        elif layer == 'encoder': act_map = feature_layers[0, index]
        act_maps.append(cv2.resize(act_map.to('cpu').detach().numpy(), (mask.shape), cv2.INTER_LINEAR))
        layers.append(layer)
        map_descriptions.append("%s - #%d, score:%.2f"%(layer, index, similarity))
        if idx >= (layer_cand_num-1): break
    return act_maps, layers, map_descriptions

def contribution_www(resnet_model, norm_inputs, target, num_top_concept, feature_layers, www_layer_list):
    all_concept_contribution = []
    all_act_maps = []
    all_concept_names = []
    sample_shap = copy.deepcopy(resnet_model)._compute_taylor_scores(norm_inputs, target)
    sample_shap = sample_shap[0][0][0,:,0,0]
    sample_shap = sample_shap.cpu().detach().numpy()
    most_important_concepts = np.argsort(sample_shap)[::-1][:num_top_concept]
    for c_id in most_important_concepts:
        heatmap = feature_layers[3][0, c_id].to('cpu').detach().numpy()
        weight = sample_shap[c_id] / np.sum(sample_shap[most_important_concepts])
        concept_name = www_layer_list[c_id][0]
        if concept_name.endswith("-c"): concept_name = concept_name[:-2]
        concept_name += ' (#%04d)'%c_id
        all_concept_contribution.append(weight)
        all_act_maps.append(heatmap)
        all_concept_names.append(concept_name)
    sort_idx = np.argsort(all_concept_contribution)[::-1]
    all_concept_contribution = [all_concept_contribution[i] for i in sort_idx]
    all_act_maps = [all_act_maps[i] for i in sort_idx]
    all_concept_names = [all_concept_names[i] for i in sort_idx]
    return all_concept_names, all_act_maps, all_concept_contribution