import json
from distutils.dir_util import copy_tree
from PIL import Image
import os
import numpy as np
import tqdm

ann_filename = "/hdd/wingrune/interactron_ade_4_steps/annotations/interactron_v2_train.json"
mask_dir = "/hdd/wingrune/interactron_ade_4_steps/train_mask"
with open(ann_filename, "r") as f:
    ann_ori = json.load(f)

dict_cats = {
    k: 0
    for k in range(150)
}
dict_cats[255] = 0
for ann in tqdm.tqdm(ann_ori['data']):
    mask = Image.open(os.path.join(mask_dir, ann["scene_name"], f"{ann['root']}.png"))
    cats = list(np.unique(mask))
    for cat in cats:
        dict_cats[cat] += 1

print(dict_cats)