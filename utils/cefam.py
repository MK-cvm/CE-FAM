import cv2
import torch
import numpy as np
import copy
import torch.nn.functional as F
import torch.nn as nn

class BaseCAM:
    def __init__(self,probe_model: torch.nn.Module):
        self.probe_model = probe_model.eval()
        self.translator = None

    def set_translator(self, translator):
        self.translator = translator.eval()

    def __call__(self, input_tensor, target_id, concept_vectors=None):
        input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
        pred_score = self.probe_model(input_tensor)
        cnn_layer_vector, feature_layers = self.probe_model.feature_extract()

        if concept_vectors is not None:
            _, concept_pred_similarity = self.translator(cnn_layer_vector, concept_vectors)
            self.probe_model.zero_grad()
            self.translator.zero_grad()
            loss = sum(concept_pred_similarity[:, target_id])
        else:
            self.probe_model.zero_grad()
            loss = sum(pred_score[:, target_id])

        loss.backward(retain_graph=True)
        grads_layers = self.probe_model.grad_extract()
        cams = []
        weights = []
        activations = []
        for layer_activation, layer_grad in zip(feature_layers, grads_layers):
            if len(feature_layers) > 1:
                layer_activation = layer_activation.cpu().data.numpy()
                weight = np.mean(layer_grad, axis=(2, 3))
            else:
                layer_activation = feature_layers.cpu().data.numpy()
                weight = np.mean(grads_layers, axis=(2, 3))              
            weighted_activations = weight[:, :, None, None] * layer_activation
            cam = weighted_activations.sum(axis=1)
            cam = np.maximum(cam, 0)
            cams.append(cam)
            weights.append(weight)
            activations.append(layer_activation)
        return cams, weights, activations
    
    def __del__(self):
        del self.probe_model, self.translator

class ProbeModel(nn.Module):
    def __init__(self, model, model_type):
        super(ProbeModel, self).__init__()
        self.model = model
        self.model_type = model_type
        self.gradient_layer4 = []
        self.gradient_layer3 = []
        self.gradient_layer2 = []
        self.gradient_layer1 = []
        self.gradient_encoder = []
        self.handles = []
        self.layer4 = []
        self.layer3 = []
        self.layer2 = []
        self.layer1 = []
        self.encoder = []
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def set_probe(self, feature=False, gradient=False):
        self.feature = feature
        self.gradient = gradient
        if self.model_type == "resnet":
            if self.feature:
                self.handles.append(self.model.layer4.register_forward_hook(self.store_feature_layer4))
                self.handles.append(self.model.layer3.register_forward_hook(self.store_feature_layer3))
                self.handles.append(self.model.layer2.register_forward_hook(self.store_feature_layer2))
                self.handles.append(self.model.layer1.register_forward_hook(self.store_feature_layer1))
            if self.gradient:
                self.handles.append(self.model.layer4.register_forward_hook(self.save_gradient_layer4))
                self.handles.append(self.model.layer3.register_forward_hook(self.save_gradient_layer3))
                self.handles.append(self.model.layer2.register_forward_hook(self.save_gradient_layer2))
                self.handles.append(self.model.layer1.register_forward_hook(self.save_gradient_layer1))
        elif self.model_type == "vit_b":
            if self.feature:
                self.handles.append(self.model.encoder.layers[-1].register_forward_hook(self.store_feature_encoder))
            if self.gradient:
                self.handles.append(self.model.encoder.layers[-1].register_forward_hook(self.save_gradient_encoder))

    def save_gradient_layer4(self, module, input, output):
        def _store_grad(grad):
            self.gradient_layer4 = [grad.cpu().detach()] + self.gradient_layer4
        output.register_hook(_store_grad)

    def save_gradient_layer3(self, module, input, output):
        def _store_grad(grad):
            self.gradient_layer3 = [grad.cpu().detach()] + self.gradient_layer3
        output.register_hook(_store_grad)

    def save_gradient_layer2(self, module, input, output):
        def _store_grad(grad):
            self.gradient_layer2 = [grad.cpu().detach()] + self.gradient_layer2
        output.register_hook(_store_grad)

    def save_gradient_layer1(self, module, input, output):
        def _store_grad(grad):
            self.gradient_layer1 = [grad.cpu().detach()] + self.gradient_layer1
        output.register_hook(_store_grad)

    def save_gradient_encoder(self, module, input, output):
        def _store_grad(grad):
            self.gradient_encoder = [grad.cpu().detach()] + self.gradient_encoder
        output.register_hook(_store_grad)

    def store_avgpool(self, module, input, output):
        self.avgpool.append(output)

    def store_feature_layer4(self, module, input, output):
        self.layer4.append(output)

    def store_feature_layer3(self, module, input, output):
        self.layer3.append(output)

    def store_feature_layer2(self, module, input, output):
        self.layer2.append(output)

    def store_feature_layer1(self, module, input, output):
        self.layer1.append(output)

    def store_feature_encoder(self, module, input, output):
        self.encoder.append(output)

    def feature_extract(self):
        if self.model_type == "resnet":
            layer4 = self.layer4[0].clone()
            layer3 = self.layer3[0].clone()
            layer2 = self.layer2[0].clone()
            layer1 = self.layer1[0].clone()
            cnn_vector4 = self.pooling(layer4)
            cnn_vector3 = self.pooling(layer3)
            cnn_vector2 = self.pooling(layer2)
            cnn_vector1 = self.pooling(layer1)
            self.layer4 = []
            self.layer3 = []
            self.layer2 = []
            self.layer1 = []
            return (cnn_vector1, cnn_vector2, cnn_vector3, cnn_vector4), (layer1, layer2, layer3, layer4)
        elif self.model_type == "vit_b":
            encoder = self.encoder[0].clone()
            encoder = self.model.encoder.ln(encoder)
            encoder = encoder[:, 1 :, :]
            encoder = encoder.reshape(encoder.size(0),14,14,encoder.size(-1))
            encoder = encoder.permute(0,3,1,2)
            encoder_vector = self.pooling(encoder)
            self.encoder = []
            return (encoder_vector), (encoder)
    
    def grad_extract(self):
        if self.model_type == "resnet":
            gradient_layer4 = self.gradient_layer4[0].cpu().data.numpy()
            gradient_layer3 = self.gradient_layer3[0].cpu().data.numpy()
            gradient_layer2 = self.gradient_layer2[0].cpu().data.numpy()
            gradient_layer1 = self.gradient_layer1[0].cpu().data.numpy()
            self.gradinet_layer4 = []
            self.gradinet_layer3 = []
            self.gradinet_layer2 = []
            self.gradinet_layer1 = []
            return (gradient_layer1, gradient_layer2, gradient_layer3, gradient_layer4)
        elif self.model_type == "vit_b":
            gradient_encoder = self.gradient_encoder[0]
            gradient_encoder = gradient_encoder[:, 1 :, :]
            gradient_encoder = gradient_encoder.reshape(gradient_encoder.size(0),14,14,gradient_encoder.size(-1))
            gradient_encoder = gradient_encoder.permute(0,3,1,2).cpu().data.numpy()
            self.gradient_encoder = []
            return (gradient_encoder)

    def forward(self, input):
        output = self.model(input)
        return output
    
    def __del__(self):
        del self.handles, self.model, self.gradient_layer4, self.gradient_layer3, self.gradient_layer2, self.gradient_layer1, self.gradient_encoder, self.layer4, self.layer3, self.layer2, self.layer1, self.encoder

def contribution_cefam(model, cnn_layer_vector, concept_vectors, concept_list, probe_layer, concept_dataset, grad_probe_model, norm_inputs, softmax, resnet_model, target, pred_score, positive_only=True, num_top_prediction=8, num_top_contribution=0):
    all_concept_contribution = []
    all_concept_names = []
    all_act_maps = []
    all_concept_maps = []
    all_scores = []
    all_del_scores = []
    all_concept_prediction = []
    top_k_num = 20
    with torch.no_grad():
        _, concept_pred_similarity = model(cnn_layer_vector, concept_vectors)
    if model.model_type == "siglip": concept_pred_similarity = torch.sigmoid(concept_pred_similarity)
    elif model.model_type == "clip": concept_pred_similarity = softmax(concept_pred_similarity)
    concept_idxes = torch.argsort(concept_pred_similarity[0], descending=True)
    if num_top_contribution == 0: concept_idxes = concept_idxes[:num_top_prediction]
    for concept_idx in concept_idxes:
        concept_name = concept_list[concept_idx][0]
        act_maps, _, _, match_index, activate_vectors, association_scores = region_cefam(model, probe_layer, grad_probe_model, norm_inputs, cnn_layer_vector, concept_vectors, concept_idx.item())
        _, concept_top_idxs = torch.topk(torch.from_numpy(activate_vectors[match_index]).clone(), k=top_k_num, dim=0)
        del_scores = []
        with torch.no_grad():
            concept_contribution = 0
            for ti in range(top_k_num+1):
                mask_score = softmax(resnet_model.mask_predict(copy.deepcopy(norm_inputs), concept_top_idxs[:ti], match_index))[0][target].item()
                del_score = (pred_score-mask_score)/pred_score
                del_scores.append(del_score)
                concept_contribution += del_score/top_k_num
        if positive_only and concept_contribution < 0: continue
        all_concept_maps.append(act_maps)
        all_concept_names.append(concept_name)
        all_act_maps.append(act_maps[match_index])
        all_scores.append(association_scores)
        all_del_scores.append(del_scores)
        all_concept_contribution.append(concept_contribution)
        all_concept_prediction.append(concept_pred_similarity[0, concept_idx].to('cpu').detach().numpy())

    top_contribution_idx = np.argsort(all_concept_contribution)[::-1][:num_top_contribution]
    top_prediction_idx = np.argsort(all_concept_prediction)[::-1][:num_top_prediction]
    top_impact_idx = np.union1d(top_contribution_idx, top_prediction_idx)
    sort_idx = top_impact_idx[np.argsort(np.abs(np.array(all_concept_contribution))[top_impact_idx])[::-1]]
    all_concept_maps = [all_concept_maps[i] for i in sort_idx]
    all_concept_names = [all_concept_names[i] for i in sort_idx]
    all_act_maps = [all_act_maps[i] for i in sort_idx]
    all_scores = [all_scores[i] for i in sort_idx]
    all_del_scores = [all_del_scores[i] for i in sort_idx]
    all_concept_contribution = [all_concept_contribution[i] for i in sort_idx]
    return all_concept_names, all_act_maps, all_concept_contribution, all_concept_maps, all_del_scores, all_scores

def region_cefam(model, probe_layer, grad_probe_model, norm_inputs, cnn_layer_vector, concept_vectors, target_id, layer_cand_num=1, mask=None):
    top_k_num = 20
    act_maps = []
    activate_vectors = []
    layers = []
    map_descriptions = []
    association_scores = []
    conceptcam = BaseCAM(probe_model=copy.deepcopy(grad_probe_model))
    conceptcam.set_translator(copy.deepcopy(model))
    concept_cams, concept_weights, concept_activations = conceptcam(input_tensor=norm_inputs, concept_vectors=concept_vectors, target_id=target_id)
    important_indexs = []
    highest_score = -1
    match_idx = -1
    cand_layer = {'all': [0,1,2,3], 'layer4':[3], 'layer3':[2], 'layer2':[1], 'layer1':[0], 'encoder':[0]}
    for layer_ind, (cam, weight, activation) in enumerate(zip(concept_cams, concept_weights, concept_activations)):
        if layer_ind not in cand_layer[probe_layer]: continue
        activate_vector = weight[0]*np.mean(activation[0], axis=(1,2))
        _, top_idxs = torch.topk(torch.from_numpy(activate_vector).clone(), k=top_k_num, dim=0)
        important_indexs.append(top_idxs)
        if mask is not None: act_map = cv2.resize(cam[0], (mask.shape), cv2.INTER_LINEAR)
        else: act_map = cam[0]
        act_maps.append(act_map)
        activate_vectors.append(activate_vector)
        layers.append('layer%d'%(layer_ind+1))
        map_descriptions.append("layer%d"%(layer_ind+1))

    if layer_cand_num == 1 and probe_layer == 'all':
        map_descriptions = []
        with torch.no_grad():
            _, concept_pred_similarity = model(copy.deepcopy(cnn_layer_vector), concept_vectors)
            if model.model_type == "siglip": concept_pred_similarity = torch.sigmoid(concept_pred_similarity)
            elif model.model_type == "clip": concept_pred_similarity = F.softmax(concept_pred_similarity, dim=-1)
            base_score = concept_pred_similarity[0, target_id].item()
            for layer_ind in range(len(concept_cams)):
                association_score = 0
                for ti in range(top_k_num+1):
                    _, concept_pred_similarity = model(copy.deepcopy(cnn_layer_vector), concept_vectors, important_indexs[layer_ind][:ti], layer_ind)
                    if model.model_type == "siglip": concept_pred_similarity = torch.sigmoid(concept_pred_similarity)
                    elif model.model_type == "clip": concept_pred_similarity = F.softmax(concept_pred_similarity, dim=-1)
                    _score = concept_pred_similarity[0, target_id].item()
                    del_score = (base_score-_score)/base_score
                    association_score += del_score/top_k_num
                association_scores.append(association_score)
                if highest_score < association_score:
                    highest_score = association_score
                    match_idx = layer_ind
                map_descriptions.append("layer%d, score:%.4f"%(layer_ind+1, association_score))
        
    return act_maps, layers, map_descriptions, match_idx, activate_vectors, association_scores