import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import re
import copy

plt.rcParams['font.family'] = 'Times New Roman'
green_rgb = np.array([0, 128, 0])/255
red_rgb = np.array([255, 6, 85])/255

def format_value(s, format_str):
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s

def contribution_bar(contributions, feature_names, save_dir, num_display=8, title=None, fontsize=24):
    num_features = min(num_display, len(contributions))
    contributions = contributions[:num_features]
    feature_names = feature_names[:num_features]
    feature_inds = np.argsort(contributions, 0)[::-1]
    y_pos = np.arange(num_features, 0, -1)
    yticklabels = []
    for i in feature_inds:
        yticklabels.append(feature_names[i])

    plt.gcf().set_size_inches(8, num_features * 0.3  + 2.0)
    negative_values_present = np.sum(contributions[feature_inds] < 0) > 0
    if negative_values_present: plt.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)
    
    plt.barh(
        y_pos, contributions[feature_inds],
        0.8, align='center',
        color=[red_rgb if contributions[feature_inds[j]] <= 0 else green_rgb for j in range(len(y_pos))],
        hatch=None, edgecolor=(1,1,1,0.8), label=None
    )

    plt.yticks(list(y_pos) + list(y_pos + 1e-8), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=fontsize)
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox_to_xscale = xlen/bbox.width

    for j in range(len(y_pos)):
        ind = feature_inds[j]
        if contributions[ind] < 0:
            plt.text(
                contributions[ind] - (5/72)*bbox_to_xscale, y_pos[j], format_value(contributions[ind], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=red_rgb,
                fontsize=fontsize
            )
        else:
            plt.text(
                contributions[ind] + (5/72)*bbox_to_xscale, y_pos[j], format_value(contributions[ind], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=green_rgb,
                fontsize=fontsize
            )

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if negative_values_present: plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params('x', labelsize=fontsize)
    xmin,xmax = plt.gca().get_xlim()
    if negative_values_present: plt.gca().set_xlim(xmin - (xmax-xmin)*0.1, xmax + (xmax-xmin)*0.1)
    else: plt.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.1)
    plt.xlabel("Contribution", fontsize=fontsize)
    fig.tight_layout() 
    plt.savefig(os.path.join(save_dir, 'contribution_bar.png'))
    plt.clf()
    plt.close()    

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "VAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg

def save_logs(model_type, save_dir, filename, concept, rgb_img, act_maps, best_act_map, mask, iou_list, lower_iou_list, upper_iou_list, vea_ratio, map_descriptions):
    fontsize = 24
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.axis("off")
    ax1.imshow(rgb_img)
    rgb_img = cv2.resize(rgb_img, mask.shape, cv2.INTER_LINEAR)
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.axis("off")
    ax2.imshow(rgb_img)
    ax2.imshow(best_act_map, cmap='jet', alpha=0.5)
    ax2.set_title("Selected Map", fontsize=fontsize)
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(rgb_img)
    ax3.imshow(mask, cmap='gray', alpha=1.0)
    ax3.set_title("Segmentation Mask", fontsize=fontsize)
    ax3.axis("off")

    for i, act_map in enumerate(act_maps):
        ax = fig.add_subplot(2, 4, i+5)
        ax.imshow(rgb_img)
        ax.imshow(act_map, cmap='jet', alpha=0.5)
        ax.set_title(map_descriptions[i], fontsize=fontsize)
        ax.axis("off")

    fig.tight_layout()  
    plt.savefig(os.path.join(save_dir, model_type, "%s_%s.png"%(filename, concept)))
    plt.clf()
    plt.close()

    fontsize = 36
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels([0, 50, 100], fontsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1], fontsize=fontsize)
    ax.set_xlabel("Top-n% Threshold", fontsize=fontsize)
    ax.set_ylabel("IoU", fontsize=fontsize)
    l1, = ax.plot(iou_list, label = 'Predicted', linewidth=3)
    l2, = ax.plot(lower_iou_list, label = 'Random', linewidth=3)
    l3, = ax.plot(upper_iou_list, label = 'Ideal', linewidth=3)
    ax.legend(handles=[l1, l2, l3], bbox_to_anchor=(0.98, 0.98), loc = 'upper right', borderaxespad=0, fontsize=fontsize-12)
    fig.tight_layout()  
    plt.savefig(os.path.join(save_dir, model_type, "%s_%s_%dVEAplot.png"%(filename, concept, vea_ratio*1000)))
    plt.clf()
    plt.close()

def save_clip_prediction(clip_data, save_dir):
    fontsize = 36
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    for idx, (element) in enumerate(clip_data):
        concept_name, similarity = element
        ax.text(0.1, 1.0-idx*0.08, "Score:%.3f  %s"%(similarity, concept_name), va='center', transform=ax.transAxes,fontsize=fontsize)
    fig.tight_layout()  
    plt.savefig(os.path.join(save_dir, "clip_prediction.png"))
    plt.clf()
    plt.close()

def save_contribution_plot(all_del_scores, all_concept_names, save_dir):
    fig = plt.figure(figsize=(6, 6))
    fontsize = 36
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 10, 20])
    ax.set_xticklabels([0, 50, 100], fontsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 50, 100], fontsize=fontsize)
    ax.set_xlabel("Masked Channels [%]", fontsize=fontsize)
    ax.set_ylabel("Decrease in Score [%]", fontsize=fontsize)
    l1, = ax.plot(all_del_scores[0], label = all_concept_names[0], linewidth=3)
    l2, = ax.plot(all_del_scores[1], label = all_concept_names[1], linewidth=3)
    l3, = ax.plot(all_del_scores[2], label = all_concept_names[2], linewidth=3)
    ax.legend(handles=[l1, l2, l3], bbox_to_anchor=(0.98, 0.85), loc = 'upper right', borderaxespad=0, fontsize=fontsize-4)
    fig.tight_layout()  
    plt.savefig(os.path.join(save_dir, "contribution_plot.png"))
    plt.clf()
    plt.close()

def save_contribution(inputs, save_dir, class_cam, all_concept_names, all_concept_contribution, all_act_maps, all_scores=None, all_concept_maps=None, num_display=8):
    contribution_bar(np.array(all_concept_contribution), np.array(all_concept_names), save_dir, num_display=num_display)
    image = inputs.permute(0, 2, 3, 1).to('cpu').detach().numpy().copy()[0]
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.axis("off")
    ax1.imshow(image)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.axis("off")
    for idx, (concept_name, contribution) in enumerate(zip(all_concept_names, all_concept_contribution)):
        ax2.text(-0.1, 1.0-idx*0.05, "%s: sim:%.3f"%(concept_name, contribution), va='center', transform=ax2.transAxes,fontsize=7)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    ax3.imshow(image)
    ax3.imshow(class_cam, cmap='jet', alpha=0.5)
    ax3.set_title("Grad-CAM", fontsize=8)
    plt.savefig(os.path.join(save_dir, "saliency_map.png"))
    plt.clf()
    plt.close()

    for i, (act_map, concept_name) in enumerate(zip(all_act_maps, all_concept_names)):
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.axis("off")
        ax1.imshow(image)
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.axis("off")
        ax2.imshow(image)
        ax2.imshow(cv2.resize(act_map, image.shape[:2], cv2.INTER_LINEAR), cmap='jet', alpha=0.5)
        ax2.set_title("Selected Map", fontsize=8)
        if all_concept_maps is not None:
            concept_maps = all_concept_maps[i]
            scores = all_scores[i]
            for ci, (concept_cam, score) in enumerate(zip(concept_maps, scores)):
                ax3 = fig.add_subplot(2, 4, ci+5)
                ax3.axis("off")
                ax3.imshow(image)
                ax3.imshow(cv2.resize(concept_cam, image.shape[:2], cv2.INTER_LINEAR), cmap='jet', alpha=0.5)
                ax3.set_title("Concept Layer%d_Score%.3f"%(ci+1, score), fontsize=8)
        plt.savefig(os.path.join(save_dir, "%s.png"%(concept_name)))
        plt.clf()
        plt.close()

class eval_logs:
    def __init__(self, method, type_log):
        self.method = method
        self.total_count = copy.deepcopy(type_log)
        self.valid_count = copy.deepcopy(type_log)
        self.hit_count = copy.deepcopy(type_log)
        self.sum_epg_score = copy.deepcopy(type_log)
        self.sum_auc_score = copy.deepcopy(type_log)
        self.sum_nra_score = copy.deepcopy(type_log)
        self.avg_epg = copy.deepcopy(type_log)
        self.avg_nra = copy.deepcopy(type_log)
        self.avg_hit_rate = copy.deepcopy(type_log)
        self.type_num = len(type_log.keys())
    def calc_epg(self, concept_type):
        return self.sum_epg_score[concept_type] / max(1, self.valid_count[concept_type])
    def calc_nra(self, concept_type):
        return self.sum_nra_score[concept_type] / max(1, self.valid_count[concept_type])
    def calc_hit_rate(self, concept_type):
        return self.hit_count[concept_type] / max(1, self.total_count[concept_type])
    def calc_average(self):
        self.epg = 0
        self.nra = 0
        self.hit_rate = 0
        for concept_type in self.valid_count.keys():
            self.avg_epg[concept_type] = self.calc_epg(concept_type)
            self.avg_nra[concept_type] = self.calc_nra(concept_type)
            self.avg_hit_rate[concept_type] = self.calc_hit_rate(concept_type)
            self.epg += self.avg_epg[concept_type] / self.type_num
            self.nra += self.avg_nra[concept_type] / self.type_num
            self.hit_rate += self.avg_hit_rate[concept_type] / self.type_num
            print("%s %s %.6f %.6f %.6f"%(self.method, concept_type, self.avg_epg[concept_type], self.avg_nra[concept_type], self.avg_hit_rate[concept_type]))
        return self.epg, self.nra, self.hit_rate