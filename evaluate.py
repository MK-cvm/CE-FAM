from __future__ import print_function
import argparse
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torchvision.transforms as torch_transforms
import pickle
import copy
import cv2
from utils.cefam import region_cefam, contribution_cefam, BaseCAM
from utils.clip_dissect import get_clip_dissect_dict, region_clip_dissect
from utils.www import get_www_dict, region_www, contribution_www
from utils.broden import get_broden_concept, make_label_list
from utils. imagenet import get_imagenet_concept, EvalDataset
from utils.log import save_logs, save_clip_prediction, save_contribution_plot, save_contribution, eval_logs
from converter import MapConverter
from utils.general import to_tensor, normalize_image, get_classifier, get_vlm
import glob
import torch.nn as nn
from utils.resnet import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from PIL import Image
from itertools import chain

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def epg_eval(act_map, mask):
    if np.sum(act_map) == 0: return 0.0
    norm_map = (act_map - np.min(act_map)) / (np.max(act_map) - np.min(act_map))
    in_value = np.sum(norm_map*mask.astype(np.float32))
    out_value = np.sum(norm_map*(~mask).astype(np.float32))
    epg_score = in_value/(in_value+out_value)
    return epg_score

def nra_eval(saliency_map, target_mask):
    image_h, image_w = saliency_map.shape
    lex_inds = np.argsort(saliency_map.ravel())[::-1]
    pixel_num = image_w*image_h
    vea_auc = 0
    upper_auc = 0
    iou_list = []
    lower_iou_list = []
    upper_iou_list = []
    mask_ratio = np.sum(target_mask) / (image_h*image_w)
    tp_sum = 0
    fp_sum = 0
    bin_num = 100
    step = image_h*image_w // bin_num
    for idx, ind in enumerate(lex_inds):
        h = ind // image_w
        w = ind % image_w
        if target_mask[h, w] == 1: tp_sum += 1
        else: fp_sum += 1
        # Considering that the mask region may be smaller than step size.
        # Compute in a fine-grained manner for sizes below the step, and coarsely for sizes above it.
        if idx < step or idx % step == 0:
            seg_iou = tp_sum / (fp_sum + target_mask.sum())
            if idx < target_mask.sum(): upper_iou = idx / target_mask.sum()
            else: upper_iou = target_mask.sum() / idx
            lower_iou = mask_ratio*idx/pixel_num
            if idx < step: unit_num = pixel_num
            else: unit_num = bin_num
            vea_auc += seg_iou/unit_num
            upper_auc += upper_iou/unit_num
            if idx % step == 0:
                upper_iou_list.append(upper_iou)
                iou_list.append(seg_iou)
                lower_iou_list.append(lower_iou)
    lower_auc = mask_ratio / 2
    nra_score = max(vea_auc-lower_auc, 0) / (upper_auc-lower_auc)
    return iou_list, lower_iou_list, upper_iou_list, nra_score

def imagenet_eval(model, probe_model, processor, probe_layer, mean, std, dir_for_output, concept_vectors, concept_type_dict, concept_list, clip_dissect_dict, www_dict, device, eval_methods, layer_cand_num, eval_images, eval_log):
    transform = torch_transforms.Compose([torch_transforms.Resize(256), torch_transforms.CenterCrop(224), torch_transforms.ToTensor()])
    concept_vectors = concept_vectors.unsqueeze(0)
    all_log = {}
    for method in eval_methods:
        all_log[method] = eval_logs(method, {'object':0.0})
        os.makedirs(os.path.join(dir_for_output, method), exist_ok=True)
    name_list=os.path.join('concepts', 'ImageNetS', 'ImageNetS_im919_validation.txt')
    gt_dir = os.path.join('../datasets','ImageNet-S', 'ImageNetS919')
    metafile = os.path.join('concepts', 'ImageNetS', 'wn_to_id.json')
    
    dataset = EvalDataset(gt_dir, name_list, transform, metafile, processor)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    feature_probe_model = copy.deepcopy(probe_model)
    feature_probe_model.set_probe(feature=True)
    grad_probe_model = copy.deepcopy(probe_model)
    grad_probe_model.set_probe(feature=True, gradient=True)
    for li, (inputs, gt_masks, target, filename) in enumerate(tqdm(val_loader)):
        filename = filename[0]
        if len(eval_images) > 0 and filename not in eval_images: continue 
        mask =gt_masks[0, 0].to('cpu').detach().numpy()
        inputs = inputs.to(device, non_blocking=True)
        norm_inputs = normalize_image(inputs, mean, std)
        with torch.no_grad():
            _ = feature_probe_model(norm_inputs)
            cnn_layer_vector, feature_layers = feature_probe_model.feature_extract()
        concept_name = concept_list[target]
        assert concept_name in concept_type_dict.keys()
        for mi, method in enumerate(eval_methods):
            concept_type = 'object'
            all_log[method].total_count[concept_type] += 1
            match_index = -1
            if method == 'CEFAM':
                act_maps, layers, map_descriptions, _, _, _ = region_cefam(model, probe_layer, grad_probe_model, norm_inputs, cnn_layer_vector, concept_vectors, target, layer_cand_num, mask)
            elif method == 'CLIP-dissect' and concept_name in clip_dissect_dict.keys():
                act_maps, layers, map_descriptions = region_clip_dissect(feature_layers, mask, clip_dissect_dict, concept_name, layer_cand_num)
            elif method == 'WWW' and concept_name in www_dict.keys():
                act_maps, layers, map_descriptions = region_www(feature_layers, mask, www_dict, concept_name, layer_cand_num)
            else: continue
            epg_score = 0
            nra_score = 0
            iou_list = None
            lower_iou_list = None
            upper_iou_list = None
            best_act_map = None
            for idx, (_act_map, _layer) in enumerate(zip(act_maps, layers)):
                if match_index != -1 and idx != match_index: continue
                _epg_score = epg_eval(_act_map, mask)
                _iou_list, _lower_iou_list, _upper_iou_list, _nra_score = nra_eval(_act_map, mask)
                if _nra_score > nra_score:
                    nra_score = _nra_score
                    epg_score = _epg_score
                    iou_list = _iou_list
                    lower_iou_list = _lower_iou_list
                    upper_iou_list = _upper_iou_list
                    best_act_map = _act_map

            all_log[method].sum_epg_score[concept_type] += epg_score
            all_log[method].sum_nra_score[concept_type] += nra_score
            all_log[method].valid_count[concept_type] += 1
            if nra_score == 0: continue
            elif nra_score > 0.5: all_log[method].hit_count[concept_type] += 1
            if eval_log:
                rgb_img = inputs.permute(0, 2, 3, 1).to('cpu').detach().numpy().copy()
                save_logs(method, dir_for_output, filename, concept_name, rgb_img[0], act_maps, best_act_map, mask, iou_list, lower_iou_list, upper_iou_list, nra_score, map_descriptions)
            if li % 50 == 0:
                epg, nra, hit_rate = all_log[method].calc_average()
                print("%s epg:%.6f nra:%.6f hitrate:%.6f"%(method, epg, nra, hit_rate))
    
    for method in eval_methods:
        epg, nra, hit_rate = all_log[method].calc_average()
        print("%s epg:%.6f nra:%.6f hitrate:%.6f"%(method, epg, nra, hit_rate))
        eval_log = all_log[method]
        with open(os.path.join(dir_for_output, method, 'imagenet_eval.txt'), 'a') as f:
            for concept_type in eval_log.valid_count.keys():
                f.write(f"type:{concept_type} total_count:{eval_log.total_count[concept_type]} valid_count:{eval_log.valid_count[concept_type]} hit_count:{eval_log.hit_count[concept_type]}\
                         epg:{eval_log.avg_epg[concept_type]} nra:{eval_log.avg_nra[concept_type]} hit_rate{eval_log.avg_hit_rate[concept_type]}\n")
            f.write(f"{method} epg:{eval_log.epg} nra:{eval_log.nra} hit_rate:{eval_log.hit_rate}\n")
            f.close()

def broden_eval(model, probe_model, mean, std, dir_for_output, 
                concept_vectors, concept_type_dict, clip_dissect_dict, www_dict, device, eval_methods, probe_layer, layer_cand_num, eval_images, eval_log):
    label_dir = '/root/work/datasets/broden1_224'
    transform = torch_transforms.transforms.Compose([torch_transforms.transforms.ToTensor()])
    label_file = os.path.join(label_dir, 'broden_concept_labels.pkl')
    if os.path.exists(label_file):
        with open(label_file, mode='br') as fi:
            label_list = pickle.load(fi)
    else:
        label_list = make_label_list(label_dir)
    concept_vectors = concept_vectors.unsqueeze(0)
    all_log = {}
    for method in eval_methods:
        all_log[method] = eval_logs(method, {'color':0.0, 'material':0.0, 'object':0.0, 'part':0.0})
        os.makedirs(os.path.join(dir_for_output, method), exist_ok=True)
    feature_probe_model = copy.deepcopy(probe_model)
    feature_probe_model.set_probe(feature=True)
    grad_probe_model = copy.deepcopy(probe_model)
    grad_probe_model.set_probe(feature=True, gradient=True)
    for li, label_dict in enumerate(tqdm(label_list)):
        image_file = label_dict['path']
        labels = label_dict['label']
        filename = os.path.splitext(os.path.basename(image_file))[-2]
        if len(eval_images) > 0 and filename not in eval_images: continue 
        image = cv2.imread(image_file)
        cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = transform(cvt_image)
        inputs = inputs.to(device, non_blocking=True).unsqueeze(0)
        norm_inputs = normalize_image(inputs, mean, std)
        with torch.no_grad():
            _ = feature_probe_model(norm_inputs)
            cnn_layer_vector, feature_layers = feature_probe_model.feature_extract()

        for mi, method in enumerate(eval_methods):
            for concept_id, (mask, concept_name, concept_type) in labels.items():
                assert concept_name in concept_type_dict.keys()
                if concept_type not in all_log[method].valid_count.keys(): continue
                all_log[method].total_count[concept_type] += 1
                match_index = -1
                if method == 'CEFAM':
                    act_maps, layers, map_descriptions, match_index, _, _ = region_cefam(model, probe_layer, grad_probe_model, norm_inputs, cnn_layer_vector, concept_vectors, concept_id-1, layer_cand_num, mask)
                elif method == 'CLIP-dissect' and concept_name in clip_dissect_dict.keys():
                    act_maps, layers, map_descriptions = region_clip_dissect(feature_layers, mask, clip_dissect_dict, concept_name, layer_cand_num)
                elif method == 'WWW' and concept_name in www_dict.keys():
                    act_maps, layers, map_descriptions = region_www(feature_layers, mask, www_dict, concept_name, layer_cand_num)
                else: continue
                epg_score = 0
                nra_score = 0
                iou_list = None
                lower_iou_list = None
                upper_iou_list = None
                best_act_map = None
                for idx, (_act_map, _layer) in enumerate(zip(act_maps, layers)):
                    if match_index != -1 and idx != match_index: continue
                    _epg_score = epg_eval(_act_map, mask)
                    _iou_list, _lower_iou_list, _upper_iou_list, _nra_score = nra_eval(_act_map, mask)
                    if _nra_score > nra_score:
                        nra_score = _nra_score
                        epg_score = _epg_score
                        iou_list = _iou_list
                        lower_iou_list = _lower_iou_list
                        upper_iou_list = _upper_iou_list
                        best_act_map = _act_map

                all_log[method].sum_epg_score[concept_type] += epg_score
                all_log[method].sum_nra_score[concept_type] += nra_score
                all_log[method].valid_count[concept_type] += 1
                if nra_score == 0: continue
                elif nra_score > 0.5: all_log[method].hit_count[concept_type] += 1
                if eval_log:
                    rgb_img = inputs.permute(0, 2, 3, 1).to('cpu').detach().numpy().copy()
                    save_logs(method, dir_for_output, filename, concept_name, rgb_img[0], act_maps, best_act_map, mask, iou_list, lower_iou_list, upper_iou_list, nra_score, map_descriptions)
            if li % 50 == 0:
                epg, nra, hit_rate = all_log[method].calc_average()
                print("%s epg:%.6f nra:%.6f hitrate:%.6f"%(method, epg, nra, hit_rate))

    for method in eval_methods:
        epg, nra, hit_rate = all_log[method].calc_average()
        print("%s epg:%.6f nra:%.6f hitrate:%.6f"%(method, epg, nra, hit_rate))
        eval_log = all_log[method]
        with open(os.path.join(dir_for_output, method, 'broden_eval.txt'), 'a') as f:
            for concept_type in eval_log.valid_count.keys():
                f.write(f"type:{concept_type} total_count:{eval_log.total_count[concept_type]} valid_count:{eval_log.valid_count[concept_type]} hit_count:{eval_log.hit_count[concept_type]}\
                         epg:{eval_log.avg_epg[concept_type]} nra:{eval_log.avg_nra[concept_type]} hit_rate{eval_log.avg_hit_rate[concept_type]}\n")
            f.write(f"{method} epg:{eval_log.epg} nra:{eval_log.nra} hit_rate:{eval_log.hit_rate}\n")
            f.close()

def contribution_eval(model, base_model, probe_model, vl_model, probe_model_name, probe_layer, imagenet_mean, imagenet_std, vl_mean, vl_std, dir_for_output, concept_vectors, concept_list, www_layer_list, concept_dataset, device, eval_methods):
    transform = torch_transforms.Compose([
                torch_transforms.Resize(256),
                torch_transforms.CenterCrop(224),
                torch_transforms.ToTensor(),
            ])
    concept_vectors = concept_vectors.unsqueeze(0)
    for model_type in eval_methods:
        os.makedirs(os.path.join(dir_for_output, model_type), exist_ok=True)

    extensions = ['*.JPG', '*.PNG', '*.JPEG']
    image_files = list(chain.from_iterable(glob.glob(f'images/{ext}') for ext in extensions))
    softmax = nn.Softmax(dim=-1) 
    feature_probe_model = copy.deepcopy(probe_model)
    feature_probe_model.set_probe(feature=True)
    grad_probe_model = copy.deepcopy(probe_model)
    grad_probe_model.set_probe(feature=True, gradient=True)
    if probe_model_name == "resnet50": resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif probe_model_name == "resnet18": resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet_model = resnet_model.to(device)
    resnet_model.eval()
    num_top_concept = 8
    for image_file in tqdm(image_files):
        image_pil = Image.open(image_file)
        filename = os.path.splitext(os.path.basename(image_file))
        image_tensor = transform(image_pil).unsqueeze(0)
        inputs = image_tensor.to(device, non_blocking=True)
        norm_inputs = normalize_image(inputs, imagenet_mean, imagenet_std)
        clip_data = []
        with torch.no_grad():
            clip_image_embeds = vl_model.get_image_features(normalize_image(inputs, vl_mean, vl_std))
            clip_image_embeds_ = clip_image_embeds / clip_image_embeds.norm(p=2, dim=-1, keepdim=True)
            sims = torch.bmm(concept_vectors, clip_image_embeds_.unsqueeze(2)) * model.scale + model.bias
            concept_similarity = sims[:, :, 0]
            concept_similarity = softmax(concept_similarity)
            clip_top_idxes = torch.argsort(concept_similarity[0], descending=True)
            for concept_idx in clip_top_idxes[:num_top_concept]:
                concept_name = concept_list[concept_idx][0]
                concept_score = concept_similarity[0][concept_idx]
                clip_data.append([concept_name, concept_score.item()])

        with torch.no_grad():
            output = feature_probe_model(norm_inputs)
            target = torch.argmax(output[0])
            pred_score = softmax(output)[0][target].item()
            cnn_layer_vector, feature_layers = feature_probe_model.feature_extract()
    
        for mi, model_type in enumerate(eval_methods):
            save_dir = os.path.join(dir_for_output, model_type, filename[0])
            os.makedirs(save_dir, exist_ok=True)
            save_clip_prediction(clip_data, save_dir)
            classcam = BaseCAM(probe_model=copy.deepcopy(grad_probe_model))
            class_cams, _, _ = classcam(input_tensor=norm_inputs, target_id=target.item())
            class_cam = cv2.resize(class_cams[-1][0], inputs.shape[2:], cv2.INTER_LINEAR)
            if model_type in ['CEFAM']:       
                all_concept_names, all_act_maps, all_concept_contribution, all_concept_maps, all_del_scores, all_scores = contribution_cefam(model, cnn_layer_vector, concept_vectors, concept_list, probe_layer, concept_dataset, grad_probe_model, norm_inputs, softmax, resnet_model, target, pred_score, num_top_prediction=num_top_concept)
                save_contribution_plot(all_del_scores, all_concept_names, save_dir)
                save_contribution(inputs, save_dir, class_cam, all_concept_names, all_concept_contribution, all_act_maps, all_scores, all_concept_maps)
            elif model_type in ['WWW']:
                all_concept_names, all_act_maps, all_concept_contribution = contribution_www(resnet_model, norm_inputs, target, num_top_concept, feature_layers, www_layer_list)
                save_contribution(inputs, save_dir, class_cam, all_concept_names, all_concept_contribution, all_act_maps)
            else: continue

def main(concept_dataset, probe_layer, probe_model_name, vl_model_name, ckpt_path, eval_methods, single_layer, layer_cand_num, eval_mode, eval_images, eval_log):
    device = torch.device("cuda:0")
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = "singlelayer%d_%s_%s_%s_%s_%d_%s_%s"%(single_layer, current_time, eval_mode, concept_dataset, probe_layer, layer_cand_num, probe_model_name, vl_model_name)
    dir_for_output = os.path.join('results', 'eval', output_dir)
    os.makedirs(dir_for_output, exist_ok=True)
    imagenet_mean = to_tensor(IMAGENET_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    imagenet_std = to_tensor(IMAGENET_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    base_model, probe_model, probe_layer_dim, num_classes = get_classifier(probe_model_name, device)
    vl_model, vl_mean, vl_std, concept_dim, tokenizer, processor, scale, bias = get_vlm(vl_model_name, device)

    concept_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'concepts')
    if concept_dataset == "broden":
        concept_num, concept_vectors, concept_list, concept_type_dict = get_broden_concept(os.path.join(concept_path,'broden'), concept_dim, tokenizer, vl_model, vl_model_name, device)        
    elif concept_dataset == "imagenet":
        concept_num, concept_vectors, concept_list, concept_type_dict = get_imagenet_concept(concept_dim, tokenizer, vl_model, vl_model_name, device)

    model=None
    clip_dissect_dict=None
    www_dict=None
    www_layer_list=None
    if 'CEFAM' in eval_methods:
        model = MapConverter(concept_dim=concept_dim, num_classes=num_classes, concept_num=concept_num, layer_dim=probe_layer_dim, scale=scale, bias=bias, model_type=vl_model_name, single_layer=single_layer).to(device)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['net'].state_dict())
    if eval_mode == 'region':
        if 'CLIP-dissect' in eval_methods:
            clip_dissect_dict = get_clip_dissect_dict(concept_dataset, probe_model_name, probe_layer, concept_type_dict)
        if 'WWW' in eval_methods:
            www_dict, www_layer_list = get_www_dict(concept_dataset, probe_model_name, probe_layer, concept_type_dict)
        if concept_dataset == "broden":
            broden_eval(model, probe_model, imagenet_mean, imagenet_std, dir_for_output, 
                        concept_vectors, concept_type_dict, clip_dissect_dict, www_dict, device, eval_methods, probe_layer, layer_cand_num, eval_images, eval_log)
        elif concept_dataset == "imagenet":
            imagenet_eval(model, probe_model, processor, probe_layer, imagenet_mean, imagenet_std, dir_for_output, concept_vectors, concept_type_dict, concept_list, clip_dissect_dict, www_dict, device, eval_methods, layer_cand_num, eval_images, eval_log)
    elif eval_mode == "contribution":
        if 'WWW' in eval_methods:
            www_dict, www_layer_list = get_www_dict(concept_dataset, probe_model_name, 'layer4', concept_type_dict)
        contribution_eval(model, base_model, probe_model, vl_model, probe_model_name, probe_layer, imagenet_mean, imagenet_std, vl_mean, vl_std, dir_for_output, concept_vectors, concept_list, www_layer_list, concept_dataset, device, eval_methods)
        
def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('--concept_dataset', default='broden', choices=['broden', 'imagenet'], type=str, dest='concept_dataset')
    parser.add_argument('--probe_layer', default='layer4', choices=['encoder', 'fc', 'layer4', 'layer3', 'layer2', 'layer1', 'all'], type=str, dest='probe_layer')
    parser.add_argument('--probe_model_name', default='resnet50', type=str, dest='probe_model_name')
    parser.add_argument('--vl_model_name', default='siglip', type=str, dest='vl_model_name')
    parser.add_argument('--ckpt_path', default='', type=str, dest='ckpt_path')
    parser.add_argument('--eval_mode', default='concept', type=str, dest='eval_mode')
    parser.add_argument('--single_layer', action='store_true')
    parser.add_argument('--eval_log', action='store_true')
    parser.add_argument('--layer_cand_num', default=4, type=int, dest='layer_cand_num')
    parser.add_argument('--eval_methods', default=['CEFAM'], nargs="*", type=str, dest='eval_methods')
    parser.add_argument('--eval_images', default=[], nargs="*", type=str, dest='eval_images')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args.concept_dataset, args.probe_layer, args.probe_model_name, args.vl_model_name, args.ckpt_path, args.eval_methods, args.single_layer, args.layer_cand_num, args.eval_mode, args.eval_images, args.eval_log)