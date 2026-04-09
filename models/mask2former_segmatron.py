import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from models.mask2former import (
    add_maskformer2_config,
)
from models.transformer import SemanticTransformer
from models.new_transformer import SemanticMultiStepTransformer
from torchtnt.utils.flops import FlopTensorDispatchMode
from utils.meta_utils import get_parameters, clone_parameters, sgd_step, set_parameters, detach_parameters, \
    detach_gradients

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

class mask2former_segmatron(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        # build Mask2Former
        self.cfg = copy.deepcopy(config)  # cfg can be modified by model

        cfg = get_cfg()

        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(self.cfg.MODEL.CONFIG)
        self.segm_model = build_model(cfg)
        self.criterion = self.segm_model.criterion
        # build fusion transformer
        if config.MODEL.TRANSFORMER == "DETR":
            self.fusion = SemanticMultiStepTransformer(config)
        elif config.MODEL.TRANSFORMER == "GPT":
            self.fusion = SemanticTransformer(config)

        fusion_parameters = count_all_parameters(self.fusion)

        print("Fusion parameters number", fusion_parameters / 1000000)

        self.logger = None
        self.mode = 'train'
        self.config = config
        self.num_actions = config.MODEL.NUM_ACTIONS
        self.fusion_times = []

        # for time measurements
        self.get_and_detach_params_times = []
        self.set_params_times = []
        self.pre_adaptive_times = []
        self.fusion_times = []
        self.segm_grad_times = []
        self.sgd_steps_times = []
        self.post_adaptive_times = []


    def predict(self, data):
        b, s, c, h, w = data["frames"].shape
        img = data["frames"].view(s, c, h, w)

        self.start_record()
        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            theta = get_parameters(self.segm_model)
        else:
            theta = get_parameters(self.segm_model.sem_seg_head)

        theta_task = detach_parameters(clone_parameters(theta))
        self.get_and_detach_params_times.append(self.stop_record())

        # get supervisor grads
        self.start_record()
        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            set_parameters(self.segm_model, theta_task)
        else:
            set_parameters(self.segm_model.sem_seg_head, theta_task)
        self.set_params_times.append(self.stop_record())

        batched_inputs = [
            {
            "image": img[i],
            "height": data["height"][0][0],
            "width": data["width"][0][0],
            }
            for i in range(self.num_actions)
        ]

        # with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.use_amp):
        # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
        # with torch.no_grad():
        if True:
            self.start_record()
            pre_adaptive_out = self.segm_model(batched_inputs)
            for k in range(len(pre_adaptive_out["image_features"])):
                pre_adaptive_out["image_features"][k] = pre_adaptive_out["image_features"][k].unsqueeze(0)
            pre_adaptive_out["embedded_memory_features"] = pre_adaptive_out["embedded_memory_features"].unsqueeze(0)
            pre_adaptive_out["mask2former_mask_features"] = pre_adaptive_out["mask2former_mask_features"].unsqueeze(0)
            pre_adaptive_out["mask_features"] = pre_adaptive_out["decoder_output"].unsqueeze(0)
            pre_adaptive_out["pred_logits"] = pre_adaptive_out["pred_logits"].unsqueeze(0)
            pre_adaptive_out["pred_masks"] = pre_adaptive_out["pred_masks"].unsqueeze(0)
            self.pre_adaptive_times.append(self.stop_record())

        self.start_record()
        fusion_out = self.fusion(pre_adaptive_out)
        learned_loss = torch.norm(fusion_out["loss"])
        self.fusion_times.append(self.stop_record())

        #print(learned_loss)
        self.start_record()
        segm_grad = torch.autograd.grad(learned_loss, theta_task, create_graph=False, retain_graph=False,
                                            allow_unused=True)
        self.segm_grad_times.append(self.stop_record())
        
        print(segm_grad)
        print(learned_loss)
        self.start_record()
        fast_weights = sgd_step(theta_task, segm_grad, self.config.MODEL.ADAPTIVE_LR)
        self.sgd_steps_times.append(self.stop_record())
        
        self.start_record()
        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            set_parameters(self.segm_model, fast_weights)
        else:
            set_parameters(self.segm_model.sem_seg_head, fast_weights)
        self.set_params_times.append(self.stop_record())

        # batched_inputs = [
        #     {
        #     "image": img[0],
        #     "height": data["height"][0][0],
        #     "width": data["width"][0][0],
        #     }
        # ]
        batched_inputs = [
            {
            "image": img[i],
            "height": data["height"][0][0],
            "width": data["width"][0][0],
            }
            for i in range(1)
        ]
        # with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.use_amp):
        # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
        with torch.inference_mode():
        # if True:
            self.start_record()
            post_adaptive_out = self.segm_model(batched_inputs)
            self.post_adaptive_times.append(self.stop_record())

        # FOR STRATEGIES STUDY ONLY
        # segm_loss = self.criterion(post_adaptive_out, data)
        self.start_record()
        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            set_parameters(self.segm_model, theta)
        else:
            set_parameters(self.segm_model.sem_seg_head, theta)
        self.set_params_times.append(self.stop_record())

        del theta_task
        post_adaptive_out["pred_logits"] = post_adaptive_out["pred_logits"].unsqueeze(0)
        post_adaptive_out["pred_masks"] = post_adaptive_out["pred_masks"].unsqueeze(0)

        # print(post_adaptive_out["pred_logits"].shape)
        # print(post_adaptive_out["pred_masks"].shape)

        if len(self.pre_adaptive_times) > 32:
            print("Pre-adaptive ", np.mean(self.pre_adaptive_times[30:]))
            print("fusion_times ", np.mean(self.fusion_times[30:]))
            print("segm_grad_times ", np.mean(self.segm_grad_times[30:]))
            print("post_adaptive_times ", np.mean(self.post_adaptive_times[30:]))
        return post_adaptive_out

    def start_record(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def stop_record(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end)

    def other_forward(self):


        batched_inputs = [
            {
            "image": img[0],
            "height": data["height"][0][0],
            "width": data["width"][0][0],
            }
        ]
        post_adaptive_out = self.segm_model(batched_inputs)

        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            set_parameters(self.segm_model, theta)
        else:
            set_parameters(self.segm_model.sem_seg_head, theta)
        del theta_task
        post_adaptive_out["pred_logits"] = post_adaptive_out["pred_logits"].unsqueeze(0)
        post_adaptive_out["pred_masks"] = post_adaptive_out["pred_masks"].unsqueeze(0)

        return post_adaptive_out

    def forward(self, data, train=True):

        b, s, c, h, w = data["frames"].shape
        img = data["frames"].view(b, s, c, h, w)

        labels = []
        for i in range(b):
            labels.append([])
            for j in range(s):
                category_ids = torch.unique(data["masks"][i][j]).type(torch.cuda.LongTensor)
                category_ids = category_ids[category_ids != 255]
                masks = torch.zeros((len(category_ids), data["height"][i][j], data["width"][i][j]))
                for k, cat in enumerate(category_ids):
                    masks[k, :, :] = (data["masks"][i][j] == cat).type(torch.cuda.DoubleTensor)
                labels[i].append({
                    "labels": category_ids,
                    "masks": masks
                })

        segm_losses = []
        supervisor_losses = []
        out_logits_list = []
        out_boxes_list = []
        

        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            theta = get_parameters(self.segm_model)
        else:
            theta = get_parameters(self.segm_model.sem_seg_head)
        batch_predicted_loss = []

        for task in range(b):
            theta_task = clone_parameters(theta)
            # get supervisor grads
            detached_theta_task = detach_parameters(theta)
            if self.cfg.MODEL.ADAPTIVE_BACKBONE:
                set_parameters(self.segm_model, detached_theta_task)
            else:
                set_parameters(self.segm_model.sem_seg_head, detached_theta_task)


            batched_inputs = [
                {
                "image": img[task][i],
                "height": data["height"][task][i],
                "width": data["width"][task][i],
                "task": data["task"][task][i]
                 }
                for i in range(self.num_actions)
            ]

            pre_adaptive_out = self.segm_model(batched_inputs)
            for k in range(len(pre_adaptive_out["image_features"])):
                pre_adaptive_out["image_features"][k] = pre_adaptive_out["image_features"][k].unsqueeze(0)
            pre_adaptive_out["embedded_memory_features"] = pre_adaptive_out["embedded_memory_features"].unsqueeze(0)
            pre_adaptive_out["mask_features"] = pre_adaptive_out["decoder_output"].unsqueeze(0)
            pre_adaptive_out["mask2former_mask_features"] = pre_adaptive_out["mask2former_mask_features"].unsqueeze(0)
            pre_adaptive_out["pred_logits"] = pre_adaptive_out["pred_logits"].unsqueeze(0)
            pre_adaptive_out["pred_masks"] = pre_adaptive_out["pred_masks"].unsqueeze(0)

            fusion_out = self.fusion(pre_adaptive_out)
            learned_loss = torch.norm(fusion_out["loss"])

            segm_grad = torch.autograd.grad(learned_loss, detached_theta_task, create_graph=True, retain_graph=True,
                                                allow_unused=True)
            fast_weights = sgd_step(detached_theta_task, segm_grad, self.config.MODEL.ADAPTIVE_LR)

            if self.cfg.MODEL.ADAPTIVE_BACKBONE:
                set_parameters(self.segm_model, fast_weights)
            else:
                set_parameters(self.segm_model.sem_seg_head, fast_weights)
            ridx = random.randint(0, 4)
            batched_inputs = [
                {
                "image": img[task][ridx],
                "height": data["height"][task][ridx],
                "width": data["width"][task][ridx],
                }
            ]
            post_adaptive_out = self.segm_model(batched_inputs)

            supervisor_loss = self.criterion(post_adaptive_out, labels[task][ridx:ridx+1])
            supervisor_losses.append({k: v.detach() for k, v in supervisor_loss.items()})       
            supervisor_loss = 2*supervisor_loss["loss_ce"] + 5 * supervisor_loss["loss_dice"] + 5 * supervisor_loss["loss_mask"]
            supervisor_loss.backward()
            # get segm grads
            fast_weights = sgd_step(theta_task, detach_gradients(segm_grad), self.config.MODEL.ADAPTIVE_LR)
            if self.cfg.MODEL.ADAPTIVE_BACKBONE:
                set_parameters(self.segm_model, fast_weights)
            else:
                set_parameters(self.segm_model.sem_seg_head, fast_weights)

            post_adaptive_out = self.segm_model(batched_inputs)
            segm_loss = self.criterion(post_adaptive_out, labels[task][ridx:ridx+1])
            segm_losses.append({k: v.detach() for k, v in segm_loss.items()})
            segm_loss = 2*segm_loss["loss_ce"] + 5 * segm_loss["loss_dice"] + 5 * segm_loss["loss_mask"]
            
            segm_loss.backward()
            batch_predicted_loss.append(learned_loss.detach())
            out_logits_list.append(post_adaptive_out["pred_logits"])
            out_boxes_list.append(post_adaptive_out["pred_masks"])


        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            set_parameters(self.segm_model, theta)
        else:
            set_parameters(self.segm_model.sem_seg_head, theta)
        predictions = {"pred_logits": torch.stack(out_logits_list, dim=0), "pred_masks": torch.stack(out_boxes_list, dim=0)}
        mean_segm_losses = {k.replace("loss", "loss_segm"):
                                    torch.mean(torch.stack([x[k] for x in segm_losses]))
                                for k, v in segm_losses[0].items()}
        mean_supervisor_losses = {k.replace("loss", "loss_supervisor"):
                                    torch.mean(torch.stack([x[k] for x in supervisor_losses]))
                                for k, v in supervisor_losses[0].items()}
        losses = mean_segm_losses
        losses.update(mean_supervisor_losses)
        losses.update({"learned loss": learned_loss})
        losses.update({"batch delta learned loss": max(batch_predicted_loss) - min(batch_predicted_loss)})
        return predictions, losses

    def eval(self):
        self.segm_model.eval()
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        self.segm_model.train(mode)
        return self

    def get_optimizer_groups(self, train_config):
        optim_groups = [
            {"params": list(self.segm_model.parameters()), "weight_decay": 0.0},
        ]
        return optim_groups

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger
