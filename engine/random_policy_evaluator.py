# ------------------------------------------------------------------------------
# Reference: https://github.com/allenai/interactron/blob/main/engine/random_policy_evaluator.py
# Modified by Tatiana Zemskova (https://github.com/wingrune)
# ------------------------------------------------------------------------------

import torchvision.ops
import numpy as np
import os
from datetime import datetime
import json
from PIL import Image
import torch
from torch.nn import functional as F

from torch.utils.data.dataloader import DataLoader

from utils.constants import THOR_CLASS_IDS, tlvis_classes
from utils.detection_utils import match_predictions_to_detections
from utils.storage_utils import collate_fn
from utils.transform_utis import transform, inv_transform
from models.detr_models.util.box_ops import box_cxcywh_to_xyxy

from datasets.multistep_dataset import MultiStepDataset

from detectron2.data import MetadataCatalog

colors = [(120, 120, 120),(180, 120, 120),
(6, 230, 230),(80, 50, 50),(4, 200, 3),(120, 120, 80),(140, 140, 140),(204, 5, 255),(230, 230, 230),(4, 250, 7),(224, 5, 255),(235, 255, 7),(150, 5, 61),(120, 120, 70),(8, 255, 51),(255, 6, 82),(143, 255, 140),(204, 255, 4),(255, 51, 7),(204, 70, 3),(0, 102, 200),(61, 230, 250),(255, 6, 51),(11, 102, 255),(255, 7, 71),(255, 9, 224),(9, 7, 230),(220, 220, 220),(255, 9, 92),(112, 9, 255),(8, 255, 214),(7, 255, 224),(255, 184, 6),(10, 255, 71),(255, 41, 10),(7, 255, 255),(224, 255, 8),(102, 8, 255),(255, 61, 6),(255, 194, 7),(255, 122, 8),(0, 255, 20),(255, 8, 41),(255, 5, 153),(6, 51, 255),(235, 12, 255),(160, 150, 20),(0, 163, 255),(140, 140, 140),(250, 10, 15),(20, 255, 0),(31, 255, 0),(255, 31, 0),(255, 224, 0),(153, 255, 0),(0, 0, 255),(255, 71, 0),(0, 235, 255),(0, 173, 255),(31, 0, 255),(11, 200, 200),(255 ,82, 0),(0, 255, 245),(0, 61, 255),(0, 255, 112),(0, 255, 133),(255, 0, 0),(255, 163, 0),(255, 102, 0),(194, 255, 0),(0, 143, 255),(51, 255, 0),(0, 82, 255),(0, 255, 41),(0, 255, 173),(10, 0, 255),(173, 255, 0),(0, 255, 153),(255, 92, 0),(255, 0, 255),(255, 0, 245),(255, 0, 102),(255, 173, 0),(255, 0, 20),(255, 184, 184),(0, 31, 255),(0, 255, 61),(0, 71, 255),(255, 0, 204),(0, 255, 194),(0, 255, 82),(0, 10, 255),(0, 112, 255),(51, 0, 255),(0, 194, 255),(0, 122, 255),(0, 255, 163),(255, 153, 0),(0, 255, 10),(255, 112, 0),(143, 255, 0),(82, 0, 255),(163, 255, 0),(255, 235, 0),(8, 184, 170),(133, 0, 255),(0, 255, 92),(184, 0, 255),(255, 0, 31),(0, 184, 255),(0, 214, 255),(255, 0, 112),(92, 255, 0),(0, 224, 255),(112, 224, 255),(70, 184, 160),(163, 0, 255),(153, 0, 255),(71, 255, 0),(255, 0, 163),(255, 204, 0),(255, 0, 143),(0, 255, 235),(133, 255, 0),(255, 0, 235),(245, 0, 255),(255, 0, 122),(255, 245, 0),(10, 190, 212),(214, 255, 0),(0, 204, 255),(20, 0, 255),(255, 255, 0),(0, 153, 255),(0, 41, 255),(0, 255, 204),(41, 0, 255),(41, 255, 0),(173, 0, 255),(0, 245, 255),(71, 0, 255),(122, 0, 255),(0, 255, 184),(0, 92, 255),(184, 255, 0),(0, 133, 255),(255, 214, 0),(25, 194, 194),(102, 255, 0),(92, 0, 255)]


class RandomPolicyEvaluator:

    def __init__(self, model, config, load_checkpoint=False):
        self.model = model
        if load_checkpoint:
            self.model.load_state_dict(
                torch.load(config.EVALUATOR.CHECKPOINT, map_location=torch.device('cpu'))['model'], strict=True)
        if config.DATASET.TEST.TYPE == "multistep":
            self.test_dataset = MultiStepDataset(config.DATASET.TEST.IMAGE_ROOT, config.DATASET.TEST.ANNOTATION_ROOT,
                                            config.DATASET.TEST.MODE, transform=transform)            
        else:
            print(f"Unknown dataset type: {config.DATASET.TEST.TYPE}")
            exit()

        self.config = config
        self.metadata = MetadataCatalog.get("ade20k_sem_seg_val")
        # take over whatever gpus are on the system
        self.device = torch.cuda.current_device()
        self.model = self.model.to(self.device)
        self.no_grad = False
        if config.MODEL.TYPE == "oneformer" or config.MODEL.TYPE == "single_frame_light":
            self.no_grad = True

        self.out_dir = config.EVALUATOR.OUTPUT_DIRECTORY + "/" + datetime.now().strftime("%m-%d-%Y-%H:%M:%S") + "/"

    def evaluate(self, save_results=False):

        # prepare data folder if we are saving
        if save_results:
            os.makedirs(self.out_dir + "images/", exist_ok=True)

        model, config = self.model, self.config.EVALUATOR
        model.eval()
        loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=1, num_workers=config.NUM_WORKERS,
                            collate_fn=collate_fn)

        conf_matrix = np.zeros((self.config.MODEL.NUM_CLASSES + 1, self.config.MODEL.NUM_CLASSES + 1), dtype=np.int64)
        predictions_time = []
        for idx, data in enumerate(loader):

            # place data on the correct device
            data["frames"] = data["frames"].to(self.device)
            data["masks"] = data["masks"].to(self.device)
            if save_results:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                if self.no_grad:
                    with torch.no_grad():
                        predictions = model.predict(data)
                else:
                    predictions = model.predict(data)
                end.record()

                # Waits for everything to finish running
                torch.cuda.synchronize()

                predictions_time.append(start.elapsed_time(end))
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                if self.no_grad:
                    with torch.no_grad():
                        predictions = model.predict(data)
                else:
                    predictions = model.predict(data)
                end.record()

                # Waits for everything to finish running
                torch.cuda.synchronize()
            if idx > 30:
                predictions_time.append(start.elapsed_time(end))            


            for b in range(predictions["pred_masks"].shape[0]):
                mask_cls_results = predictions["pred_logits"][b][0]
                mask_pred_results = predictions["pred_masks"][b][0].unsqueeze(0)

                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(data["height"][b][0], data["width"][b][0]),
                    mode="bilinear",
                    align_corners=False,
                )
                mask_pred_results = mask_pred_results.squeeze(0)
                semseg = model.segm_model.semantic_inference(mask_cls_results, mask_pred_results)
                semseg = semseg.argmax(dim=0).cpu()
                pred = np.array(semseg, dtype=np.int64)
                gt = data["masks"][b][0].cpu().numpy().astype(np.int64)
                gt[gt == 255] = self.config.MODEL.NUM_CLASSES

                if save_results:
                    img = Image.fromarray(np.moveaxis(data["frames"][0][0].cpu().numpy()[:,:,:240], 0, -1).astype(np.uint8))
                    img.save(f"{self.out_dir}/images/frame_{idx}.png")
                    img = Image.fromarray(np.moveaxis(data["frames"][0][1].cpu().numpy()[:,:,:240], 0, -1).astype(np.uint8))
                    img.save(f"{self.out_dir}/images/frame_add_{idx}.png")
                    img = Image.fromarray(gt.astype(np.uint8)[:,:240]).convert('P')
                    img.putpalette(np.array(colors, dtype=np.uint8))
                    img.save(f"{self.out_dir}/images/gt_{idx}.png")
                    img = Image.fromarray(pred.astype(np.uint8)[:,:240]).convert('P')
                    img.putpalette(np.array(colors, dtype=np.uint8))
                    img.save(f"{self.out_dir}/images/pred_{idx}.png")
                conf_matrix += np.bincount(
                    (self.config.MODEL.NUM_CLASSES + 1) * pred.reshape(-1) + gt.reshape(-1),
                    minlength=conf_matrix.size,
                ).reshape(conf_matrix.shape)
            del predictions

        acc = np.full(self.config.MODEL.NUM_CLASSES, np.nan, dtype=np.float)
        iou = np.full(self.config.MODEL.NUM_CLASSES, np.nan, dtype=np.float)
        tp = conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(model.segm_model.metadata.stuff_classes):
            if name != 'unlabeled':
                res[f"IoU-{name}"] = 100 * iou[i]

        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(model.segm_model.metadata.stuff_classes):
            if name != 'unlabeled':
                res[f"ACC-{name}"] = 100 * acc[i]

        print("mIoU: ",  res["mIoU"], "fwIoU: ", res["fwIoU"], "mACC: ", res["mACC"], "pACC: ", res["pACC"])
        print("Mean time, spent on inference:", np.mean(predictions_time))
        print("Std time, spent on inference:", np.std(predictions_time))
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.out_dir + "results.json", 'w') as f:
            json.dump(res, f)

        return res["mIoU"], res["fwIoU"], res["mACC"], res["pACC"]
