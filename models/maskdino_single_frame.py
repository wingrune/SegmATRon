import torch
import torch.nn as nn
import copy
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.modeling import build_model

from models.maskdino import (
    add_maskdino_config,
)
from models.maskdino.utils import box_ops
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import BitMasks, Boxes, Instances


class maskdino_single_frame(nn.Module):

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

        self.logger = None
        self.mode = 'train'
        self.config = config

    def predict(self, data):

        # reformat img data

        b, s, c, h, w = data["frames"].shape
        img = data["frames"].view(s, c, h, w)

        batched_inputs = [
            {
                "image": img[0],
                "height": data["height"][0][0],
                "width": data["width"][0][0],
            }
        ]

        out = self.segm_model(batched_inputs)

        out["pred_logits"] = out["pred_logits"].unsqueeze(0)
        out["pred_masks"] = out["pred_masks"].unsqueeze(0)

        return out

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
        out_logits_list = []
        out_masks_list = []

        for task in range(b):
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

            out = self.segm_model(batched_inputs)
            segm_loss = self.criterion(out, labels[task][ridx:ridx+1], out["mask_dict"])
            segm_losses.append({k: v.detach() for k, v in segm_loss.items()})
            segm_loss = 4*segm_loss["loss_ce"] + 5 * segm_loss["loss_dice"] + \
                5 * segm_loss["loss_mask"] + \
                5 * segm_loss['loss_bbox'] + 2 * segm_loss['loss_giou']
            segm_loss.backward()

            out_logits_list.append(out["pred_logits"])
            out_masks_list.append(out["pred_masks"])

        predictions = {
            "pred_logits": torch.stack(out_logits_list, dim=0),
            "pred_masks": torch.stack(out_masks_list, dim=0)
        }
        mean_segm_losses = {
            k.replace("loss", "loss_segm"):
            torch.mean(torch.stack([x[k] for x in segm_losses]))
            for k, v in segm_losses[0].items()
        }

        losses = mean_segm_losses

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