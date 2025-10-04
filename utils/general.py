import os
import torch
import torchvision.models as models
from utils.cefam import ProbeModel
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import numpy as np

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]

def to_tensor(x):
    x = np.array(x)
    x = torch.from_numpy(x).float()
    return x

def normalize_image(x, mean, std):
    norm_x = (x - mean) / std
    return norm_x

def get_classifier(probe_model_name, device):
    num_classes = 1000
    if probe_model_name == 'resnet50':
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        base_model.eval().to(device)
        probe_model = ProbeModel(base_model, "resnet")
        probe_layer_dim=(256,512,1024,2048)
    elif probe_model_name == 'resnet18':
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base_model.eval().to(device)
        probe_model = ProbeModel(base_model, "resnet")
        probe_layer_dim=(64,128,256,512)
    elif probe_model_name == 'vit_b_16':
        base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        base_model.eval().to(device)
        probe_model = ProbeModel(base_model, "vit_b")
        probe_layer_dim=(768,768,768,768)
    return base_model, probe_model, probe_layer_dim, num_classes

def get_vlm(vl_model_name, device):
    if vl_model_name == "siglip":
        vl_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        scale = vl_model.logit_scale.exp().item()
        bias = vl_model.logit_bias.item()
        concept_dim = 768
        vl_mean = to_tensor(SIGLIP_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
        vl_std = to_tensor(SIGLIP_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    elif vl_model_name == "clip":
        vl_model = AutoModel.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
        scale = vl_model.logit_scale.exp().item()
        bias = 0
        concept_dim = 512
        vl_mean = to_tensor(CLIP_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
        vl_std = to_tensor(CLIP_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    vl_model.eval()
    vl_model.to(device)
    return vl_model, vl_mean, vl_std, concept_dim, tokenizer, processor, scale, bias