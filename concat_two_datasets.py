import json
from distutils.dir_util import copy_tree
import os
import tqdm

ann_filename = "/hdd/wingrune/interactron_ade_4_steps/annotations/interactron_v1_train-Copy2.json"
ann_filename_to_add =  "/hdd/wingrune/interactron_ade_4_steps_add/annotations/interactron_v1_train.json"
ann_filename_save = "/hdd/wingrune/interactron_ade_4_steps/annotations/interactron_v2_train.json"

image_ori_path = "/hdd/wingrune/interactron_ade_4_steps/train"
mask_ori_path = "/hdd/wingrune/interactron_ade_4_steps/train_mask"

image_add_path = "/hdd/wingrune/interactron_ade_4_steps_add/train"
mask_add_path = "/hdd/wingrune/interactron_ade_4_steps_add/train_mask"

with open(ann_filename, "r") as f:
    ann_ori = json.load(f)

with open(ann_filename_to_add, "r") as f:
    ann_to_add = json.load(f)

for ann in tqdm.tqdm(ann_to_add["data"]):
    new_scene_name = f"{ann['scene_name']}_add"
    #copy_tree(os.path.join(image_add_path, ann['scene_name']), os.path.join(image_ori_path, new_scene_name))
    #copy_tree(os.path.join(mask_add_path, ann['scene_name']), os.path.join(mask_ori_path, new_scene_name))
    ann['scene_name'] = new_scene_name
    ann_ori['data'].append(ann)

with open(ann_filename_save, "w") as f:
    json.dump(ann_ori, f)