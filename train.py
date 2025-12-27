from __future__ import print_function
import argparse
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as torch_transforms
from utils.general import to_tensor, normalize_image
from utils.log import AverageMeter, log_msg
from utils.imagenet import get_imagenet_concept
from utils.broden import get_broden_concept
from utils.cefam import ProbeModel
from utils.lr_scheduler import WarmupReduceLROnPlateauScheduler
from transformers import AutoModel, AutoTokenizer
from converter import MapConverter
import gc

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]

def main(data_dir, max_epochs, train_batch, test_batch, vl_model_name, score_loss_mode, single_layer_mode, init_lr, probe_model_type, concept_name, warmup_epoch, patience):
    device = torch.device("cuda:0")

    if vl_model_name == "siglip":
        vl_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        scale = vl_model.logit_scale.exp().item()
        bias = vl_model.logit_bias.item()
        concept_dim = 768
    elif vl_model_name == "clip":
        vl_model = AutoModel.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        scale = vl_model.logit_scale.exp().item()
        bias = 0
        concept_dim = 512
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    vl_model.eval()
    vl_model.to(device)
    concept_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'concepts')

    if concept_name == 'broden':
        concept_num, concept_vectors, concept_list, concept_type_dict = get_broden_concept(os.path.join(concept_path,'broden'), concept_dim, tokenizer, vl_model, vl_model_name, device)
        concept_names = [concept_name for concept_name, concept_type in concept_list]
    elif concept_name == 'imagenet':
        concept_num, concept_vectors, concept_list, concept_type_dict = get_imagenet_concept(concept_dim, tokenizer, vl_model, vl_model_name, device)
        concept_names = [concept_name for concept_name in concept_list]
    train_concept_vectors = concept_vectors.unsqueeze(0).repeat(train_batch, 1, 1)
    test_concept_vectors = concept_vectors.unsqueeze(0).repeat(test_batch, 1, 1)

    label_list = []
    with open(os.path.join(concept_path, 'ImageNetS', 'imagenet_classes.txt')) as f:
        label_list = f.read().splitlines()
    if probe_model_type == 'resnet50':
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model_arch = "resnet"
        probe_layer_dim = (256,512,1024,2048)
    elif probe_model_type == 'resnet18':
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model_arch = "resnet"
        probe_layer_dim = (64,128,256,512)
    elif probe_model_type == 'vit_b_16':
        base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model_arch = "vit_b"
        probe_layer_dim = (768,768,768,768)
    num_classes = len(label_list)
    base_model.eval().to(device)
    probe_model = ProbeModel(base_model, model_arch)

    transform_train = torch_transforms.Compose([
                torch_transforms.RandomResizedCrop(224),
                torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor(),
            ])
    transform_val = torch_transforms.Compose([
                torch_transforms.Resize(256),
                torch_transforms.CenterCrop(224),
                torch_transforms.ToTensor(),
            ])
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_val)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    dir_for_output = "results/" + current_time + "_%s_%s_%s_scoreloss%d_singlelayer%d_batch%d"%(vl_model_name,probe_model_type,concept_name,score_loss_mode,single_layer_mode,train_batch)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=test_batch, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    model = MapConverter(concept_dim=concept_dim, num_classes=num_classes, concept_num=concept_num, layer_dim=probe_layer_dim, scale=scale, bias=bias, model_type=vl_model_name, single_layer=single_layer_mode).to(device)
    start_epoch = 0
    best_loss = 1000
    best_loss_embeds = 1000
    best_loss_score = 1000
    best_epoch = 0

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    scheduler = WarmupReduceLROnPlateauScheduler(
        optimizer, 
        init_lr=init_lr, 
        peak_lr=init_lr*(warmup_epoch+1), 
        warmup_steps=warmup_epoch, 
        patience=patience,
        factor=0.1,
    )

    print("The number of concept is %d"%concept_num)
    os.makedirs(dir_for_output, exist_ok=True)
    with open(os.path.join(dir_for_output, 'loss_record.txt'), 'a') as f:
        f.write("epoch train_loss train_loss_embeds train_loss_score test_loss test_loss_embeds test_loss_score\n")
        f.close()

    probe_model.set_probe(feature=True)
    for epoch in range(start_epoch, max_epochs):
        train_loss, train_loss_embeds, train_loss_score = train(train_loader, loss_fn, train_concept_vectors, model, optimizer, vl_model, probe_model, epoch, device, score_loss_mode)
        test_loss, test_loss_embeds, test_loss_score = test(val_loader, loss_fn, test_concept_vectors, model, concept_names, vl_model, probe_model, dir_for_output, epoch, device, score_loss_mode)

        with open(os.path.join(dir_for_output, 'loss_record.txt'), 'a') as f:
            f.write("%d %.10f %.10f %.10f %.10f %.10f %.10f\n" % (epoch, train_loss, train_loss_embeds, train_loss_score, test_loss, test_loss_embeds, test_loss_score))
            f.close()
        if test_loss < best_loss:
            best_loss = test_loss
            best_loss_embeds = test_loss_embeds
            best_loss_score = test_loss_score
            best_epoch = epoch
            print('NewBest, epoch = %d, loss = %.6f' % (best_epoch, best_loss))
            state = {
                'net': model,
                'loss': best_loss,
                'loss_embeds': best_loss_embeds,
                'loss_score': best_loss_score,
                'epoch': best_epoch,
            }
            torch.save(state, os.path.join(dir_for_output, 'best_converter_ckpt.pth'))
        print('Best Epoch:%d, Loss:%.10f, LossScore:%.10f' % (best_epoch, best_loss, best_loss_score))
        if (epoch - best_epoch) >= 10: break

        if epoch < warmup_epoch:
            lr = scheduler.step()
        else:
            with open(os.path.join(dir_for_output, 'loss_record.txt')) as f:
                lines = f.readlines()
                record_loss = float(lines[epoch+1].split()[4])
                lr = scheduler.step(record_loss)
        print("next learning rate: %.5f"%(lr))

def train(train_loader, loss_fn, concept_vectors, model, optimizer, vl_model, probe_model, epoch, device, score_loss_mode):

    imagenet_mean = to_tensor(IMAGENET_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    imagenet_std = to_tensor(IMAGENET_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    if model.model_type == "siglip":
        vl_mean = to_tensor(SIGLIP_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
        vl_std = to_tensor(SIGLIP_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    elif model.model_type == "clip":
        vl_mean = to_tensor(CLIP_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
        vl_std = to_tensor(CLIP_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    nb = len(train_loader)
    pbar = enumerate(train_loader)
    avg_loss = 0
    avg_loss_embeds = 0
    avg_loss_score = 0

    train_meters = {
        "losses": AverageMeter(),
        "losses_embeds": AverageMeter(),
        "losses_score": AverageMeter(),
    }
    pbar = tqdm(pbar, total=nb)
    optimizer.zero_grad()
    model.train()

    for inputs, targets in train_loader:
        inputs_cuda = inputs.to(device, non_blocking=True)
        norm_inputs = normalize_image(inputs_cuda, imagenet_mean, imagenet_std)
        with torch.no_grad():
            clip_image_embeds = vl_model.get_image_features(normalize_image(inputs_cuda, vl_mean, vl_std))
            clip_image_embeds_ = clip_image_embeds / clip_image_embeds.norm(p=2, dim=-1, keepdim=True)
            sims = torch.bmm(concept_vectors, clip_image_embeds_.unsqueeze(2)) * model.scale + model.bias
            concept_similarity = sims[:, :, 0]
            probe_model(norm_inputs)
            cnn_layer_vector, _ = probe_model.feature_extract()

        cnn_layer_embeds, concept_pred_similarity = model(cnn_layer_vector, concept_vectors)
        loss_embeds = loss_fn(clip_image_embeds, cnn_layer_embeds)
        loss_score = loss_fn(concept_pred_similarity, concept_similarity)
        loss_score *= 0.001
        if score_loss_mode: loss = loss_embeds + loss_score
        else: loss = loss_embeds
        
        train_meters["losses"].update(loss.item(), inputs.shape[0])
        train_meters["losses_embeds"].update(loss_embeds.item(), inputs.shape[0])
        train_meters["losses_score"].update(loss_score.item(), inputs.shape[0])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        msg = "Epoch:{}|Loss:{:.8f}|Emb:{:.8f}|Score:{:.8f}".format(epoch, train_meters["losses"].avg, train_meters["losses_embeds"].avg, train_meters["losses_score"].avg)
        pbar.set_description(log_msg(msg, "TRAIN"))
        pbar.update()

    pbar.close()
    avg_loss = train_meters['losses'].avg
    avg_loss_embeds = train_meters['losses_embeds'].avg
    avg_loss_score = train_meters['losses_score'].avg
    return avg_loss, avg_loss_embeds, avg_loss_score

def test(val_loader, loss_fn, concept_vectors, model, concept_names, vl_model, probe_model, dir_for_output, epoch, device, score_loss_mode, create_map=False):
    model.eval()
    imagenet_mean = to_tensor(IMAGENET_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    imagenet_std = to_tensor(IMAGENET_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    if model.model_type == "siglip":
        vl_mean = to_tensor(SIGLIP_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
        vl_std = to_tensor(SIGLIP_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    elif model.model_type == "clip":
        vl_mean = to_tensor(CLIP_MEAN).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
        vl_std = to_tensor(CLIP_STD).to(device, non_blocking=True).unsqueeze(1).unsqueeze(2)
    avg_loss = 0
    avg_loss_embeds = 0
    avg_loss_score = 0
    val_meters = {
        "losses": AverageMeter(),
        "losses_embeds": AverageMeter(),
        "losses_score": AverageMeter(),
    }
    pbar = tqdm(range(len(val_loader)))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device, non_blocking=True)
        norm_inputs = normalize_image(inputs, imagenet_mean, imagenet_std)
        with torch.no_grad():
            clip_image_embeds = vl_model.get_image_features(normalize_image(inputs, vl_mean, vl_std))
            clip_image_embeds_ = clip_image_embeds / clip_image_embeds.norm(p=2, dim=-1, keepdim=True)
            sims = torch.bmm(concept_vectors, clip_image_embeds_.unsqueeze(2)) * model.scale + model.bias
            concept_similarity = sims[:, :, 0]
            probe_model(norm_inputs)
            cnn_layer_vector, _ = probe_model.feature_extract()

            cnn_layer_embeds, concept_pred_similarity = model(cnn_layer_vector, concept_vectors)
            loss_embeds = loss_fn(clip_image_embeds, cnn_layer_embeds)
            loss_score = loss_fn(concept_pred_similarity, concept_similarity)
            loss_score *= 0.001
            if score_loss_mode: loss = loss_embeds + loss_score
            else: loss = loss_embeds
            val_meters["losses"].update(loss.item(), inputs.shape[0])
            val_meters["losses_embeds"].update(loss_embeds.item(), inputs.shape[0])
            val_meters["losses_score"].update(loss_score.item(), inputs.shape[0])
            msg = "Epoch:{}|Loss:{:.8f}|Emb:{:.8f}|Score:{:.8f}".format(epoch, val_meters["losses"].avg, val_meters["losses_embeds"].avg, val_meters["losses_score"].avg)
            pbar.set_description(log_msg(msg, "VAL"))
            pbar.update()
            del concept_pred_similarity, cnn_layer_vector, cnn_layer_embeds, clip_image_embeds, sims, norm_inputs, loss, loss_embeds, loss_score
            gc.collect()
    pbar.close()
    avg_loss = val_meters['losses'].avg
    avg_loss_embeds = val_meters['losses_embeds'].avg
    avg_loss_score = val_meters['losses_score'].avg
    return avg_loss, avg_loss_embeds, avg_loss_score

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('--data_dir', default='/root/work/datasets/ILSVRC2012/', type=str, dest='data_dir')
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--warmup_epoch', default=1, type=int)
    parser.add_argument('--patience', default=6, type=int)
    parser.add_argument('--train_batch', default=128, type=int)
    parser.add_argument('--test_batch', default=256, type=int)
    parser.add_argument('--concept_name', default='broden', type=str)
    parser.add_argument('--probe_model', default='resnet50', type=str)
    parser.add_argument('--model_type', default='', type=str)
    parser.add_argument('--score_loss', action='store_true')
    parser.add_argument('--single_layer', action='store_true')
    parser.add_argument('--init_lr', default=0.1, type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args.data_dir, args.epochs, args.train_batch, args.test_batch, args.model_type, args.score_loss, args.single_layer, args.init_lr, args.probe_model, args.concept_name, args.warmup_epoch, args.patience)
