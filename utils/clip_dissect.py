import os
import csv
import cv2

def get_clip_dissect_dict(concept_dataset, probe_model_name, probe_layer, concept_type_dict):
    with open(os.path.join('concepts', 'CLIP-dissect', probe_model_name, concept_dataset, 'descriptions.csv')) as f:
        reader = csv.reader(f, delimiter=',')
        concept_layer_dict = {}
        for i, row in enumerate(reader):
            if i == 0: continue
            layer, unit, description, similarity = row
            if layer == 'conv1' or (probe_layer not in [layer, 'all']): continue
            if description.endswith("-c"): description = description[:-2]
            assert description in concept_type_dict.keys()
            if description not in concept_layer_dict.keys():
                concept_layer_dict[description] = [(float(similarity), layer, int(unit))]
            else:
                concept_layer_dict[description].append((float(similarity), layer, int(unit)))
    for key, value in concept_layer_dict.items():
        value.sort(key=lambda x: x[0], reverse=True)
    return concept_layer_dict

def region_clip_dissect(feature_layers, mask, clip_dissect_dict, concept_name, layer_cand_num):
    act_maps = []
    layers = []
    map_descriptions = []
    for idx, (similarity, layer, index) in enumerate(clip_dissect_dict[concept_name]):
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