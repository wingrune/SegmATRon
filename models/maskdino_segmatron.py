import torch
import torch.nn as nn
import random
import copy
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.modeling import build_model

from models.maskdino import (
    add_maskdino_config,
)
from models.maskdino.utils import box_ops
from detectron2.projects.deeplab import add_deeplab_config
from models.transformer import SemanticTransformer
from utils.meta_utils import get_parameters, clone_parameters, sgd_step, set_parameters, detach_parameters, \
    detach_gradients


class maskdino_segmatron(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        # build MaskDINO
        self.cfg = copy.deepcopy(config)  # cfg can be modified by model

        cfg = get_cfg()

        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(self.cfg.MODEL.CONFIG)
        self.segm_model = build_model(cfg)
        self.criterion = self.segm_model.criterion
        # build fusion transformer
        # self.fusion = SemanticMultiStepTransformer(config)
        if config.MODEL.TRANSFORMER == "DETR":
            self.fusion = SemanticMultiStepTransformer(config)
        elif config.MODEL.TRANSFORMER == "GPT":
            self.fusion = SemanticTransformer(config)
        self.logger = None
        self.mode = 'train'
        self.config = config
        self.num_actions = config.MODEL.NUM_ACTIONS

    def predict(self, data):

        # reformat img data

        b, s, c, h, w = data["frames"].shape
        img = data["frames"].view(s, c, h, w)

        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            theta = get_parameters(self.segm_model)
        else:
            theta = get_parameters(self.segm_model.sem_seg_head)

        theta_task = detach_parameters(clone_parameters(theta))

        # get supervisor grads
        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            set_parameters(self.segm_model, theta_task)
        else:
            set_parameters(self.segm_model.sem_seg_head, theta_task)

        batched_inputs = [
            {
            "image": img[i],
            "height": data["height"][0][0],
            "width": data["width"][0][0],
            }
            for i in range(self.num_actions)
        ]

        pre_adaptive_out = self.segm_model(batched_inputs)
        pre_adaptive_out["embedded_memory_features"] = pre_adaptive_out["embedded_memory_features"].unsqueeze(0)
        pre_adaptive_out["mask_features"] = pre_adaptive_out["decoder_output"].unsqueeze(0)
        pre_adaptive_out["pred_logits"] = pre_adaptive_out["pred_logits"].unsqueeze(0)
        pre_adaptive_out["pred_masks"] = pre_adaptive_out["pred_masks"].unsqueeze(0)
        pre_adaptive_out["maskdino_mask_features"] = pre_adaptive_out["maskdino_mask_features"].unsqueeze(0)

        fusion_out = self.fusion(pre_adaptive_out)
        learned_loss = torch.norm(fusion_out["loss"])
        segm_grad = torch.autograd.grad(learned_loss, theta_task, create_graph=False, retain_graph=False,
                                            allow_unused=True)
        fast_weights = sgd_step(theta_task, segm_grad, self.config.MODEL.ADAPTIVE_LR)
        if self.cfg.MODEL.ADAPTIVE_BACKBONE:
            set_parameters(self.segm_model, fast_weights)
        else:
            set_parameters(self.segm_model.sem_seg_head, fast_weights)

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
        image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device="cuda")
        labels = []
        for i in range(b):
            labels.append([])
            for j in range(s):
                category_ids = torch.unique(data["masks"][i][j]).type(torch.cuda.LongTensor)
                category_ids = category_ids[category_ids != 255]
                masks = torch.zeros((len(category_ids), data["height"][i][j], data["width"][i][j]))
                for k, cat in enumerate(category_ids):
                    masks[k, :, :] = (data["masks"][i][j] == cat).type(torch.cuda.FloatTensor)
                boxes = torch.zeros(masks.shape[0], 4, dtype=torch.float32)
                x_any = torch.any(masks, dim=1)
                y_any = torch.any(masks, dim=2)
                for idx in range(masks.shape[0]):
                    x = torch.where(x_any[idx, :])[0]
                    y = torch.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        boxes[idx, :] = torch.as_tensor(
                            [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                        )
                labels[i].append({
                    "labels": category_ids,
                    "masks": masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(boxes.cuda())/image_size_xyxy
                })
        segm_losses = []
        supervisor_losses = []
        out_logits_list = []
        out_masks_list = []

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
                "instances": labels[task][i],
                "width": data["width"][task][i],
                "task": data["task"][task][i]
                 }
                for i in range(self.num_actions)
            ]

            pre_adaptive_out = self.segm_model(batched_inputs)
            pre_adaptive_out["embedded_memory_features"] = pre_adaptive_out["embedded_memory_features"].unsqueeze(0)
            pre_adaptive_out["mask_features"] = pre_adaptive_out["decoder_output"].unsqueeze(0)
            pre_adaptive_out["maskdino_mask_features"] = pre_adaptive_out["maskdino_mask_features"].unsqueeze(0)
            pre_adaptive_out["pred_logits"] = pre_adaptive_out["pred_logits"].unsqueeze(0)
            pre_adaptive_out["pred_masks"] = pre_adaptive_out["pred_masks"].unsqueeze(0)

            fusion_out = self.fusion(pre_adaptive_out)
            learned_loss = torch.norm(fusion_out["loss"])

            print("Learned loss", learned_loss)

            segm_grad = torch.autograd.grad(learned_loss, detached_theta_task, create_graph=True, retain_graph=True,
                                                allow_unused=True)
            fast_weights = sgd_step(detached_theta_task, segm_grad, self.config.MODEL.ADAPTIVE_LR)

            if self.cfg.MODEL.ADAPTIVE_BACKBONE:
                set_parameters(self.segm_model, fast_weights)
            else:
                set_parameters(self.segm_model.sem_seg_head, fast_weights)

            import random
            ridx = random.randint(0, 4)
            batched_inputs = [
                {
                    "image": img[task][ridx],
                    "instances": labels[task][ridx],
                    "height": data["height"][task][ridx],
                    "width": data["width"][task][ridx],
                }
            ]

            post_adaptive_out = self.segm_model(batched_inputs)

            supervisor_loss = self.criterion(post_adaptive_out, labels[task][ridx:ridx+1], post_adaptive_out["mask_dict"])
            supervisor_losses.append({k: v.detach() for k, v in supervisor_loss.items()})       
            supervisor_loss = 4*supervisor_loss["loss_ce"] + 5 * supervisor_loss["loss_dice"] + \
                    5 * supervisor_loss["loss_mask"] + \
                    5 * supervisor_loss['loss_bbox'] + 2 * supervisor_loss['loss_giou']
            supervisor_loss.backward()
            # get segm grads
            fast_weights = sgd_step(theta_task, detach_gradients(segm_grad), self.config.MODEL.ADAPTIVE_LR)
            if self.cfg.MODEL.ADAPTIVE_BACKBONE:
                set_parameters(self.segm_model, fast_weights)
            else:
                set_parameters(self.segm_model.sem_seg_head, fast_weights)

            out = self.segm_model(batched_inputs)
            segm_loss = self.criterion(out, labels[task][ridx:ridx+1], out["mask_dict"])
            segm_losses.append({k: v.detach() for k, v in segm_loss.items()})
            segm_loss = 4 * segm_loss["loss_ce"] + 5 * segm_loss["loss_dice"] + \
                5 * segm_loss["loss_mask"] + \
                5 * segm_loss['loss_bbox'] + 2 * segm_loss['loss_giou']

            segm_loss.backward()
            batch_predicted_loss.append(learned_loss.detach())
            out_logits_list.append(out["pred_logits"])
            out_masks_list.append(out["pred_masks"])

        set_parameters(self.segm_model, theta)
        predictions = {"pred_logits": torch.stack(out_logits_list, dim=0), "pred_masks": torch.stack(out_masks_list, dim=0)}
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
