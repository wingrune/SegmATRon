from utils.config_utils import (
    get_config,
    get_args,
    build_model,
    build_evaluator
)

from fvcore.nn import FlopCountAnalysis
import torch
import random
import copy
import numpy as np
import json
from torch.nn import functional as F
from torchtnt.utils.flops import FlopTensorDispatchMode

config_file = "configs/mask2former/segmatron_mask2former_4_steps_r50_habitat.yaml"

cfg = get_config(config_file)

model = build_model(cfg)
model = model.cuda()
model = model.eval()

masks = []
widths = []
heights = []
tasks = []
frames = []

for i in range(5):
    # load image
    image = np.random.random((240, 320, 3))

    sem_seg_gt = np.random.random((240, 320))
    # Pad image and segmentation label here!
    image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    if sem_seg_gt is not None:
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

    image_size = (image.shape[-2], image.shape[-1])
    padding_size = [
        0,
        320 - image_size[1],
        0,
        320 - image_size[0],
    ]
    image = F.pad(image, padding_size, value=128).contiguous()
    if sem_seg_gt is not None:
        sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=255).contiguous()

    image_shape = (image.shape[-2], image.shape[-1]) # h, w

    masks.append(sem_seg_gt)
    task = "The task is semantic"

    widths.append(image_shape[1])
    heights.append(image_shape[0])
    tasks.append(task)
    frames.append(image)

batch = [{
    'frames': frames,
    "masks": masks,
    "actions": [0, 0, 0, 0],
    "height": heights,
    "width": widths,
    "task": tasks,
    "episode_ids": 0,
    "initial_image_path": "dummy"
}]

collated_batch = {
    'frames': torch.stack([torch.stack(b['frames']) for b in batch]).float().cuda(),
    "masks": torch.stack([torch.stack(b['masks']) for b in batch]).float().cuda(),
    "actions": torch.stack([torch.tensor(b['actions'], dtype=torch.long) for b in batch]).float().cuda(),
    "episode_ids": torch.stack([torch.tensor(b['episode_ids'], dtype=torch.long) for b in batch]).float().cuda(),
    "initial_image_path": [b['initial_image_path'] for b in batch],
    "height": [b['height'] for b in batch],
    "width": [b['width'] for b in batch],
    "task": [b['task'] for b in batch],
}

for i in range(1000):
    out = model.predict(collated_batch)
