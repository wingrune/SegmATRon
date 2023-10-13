# ------------------------------------------------------------------------------
# Reference: https://github.com/allenai/interactron/blob/main/datasets/sequence_dataset.py
# Modified by Tatiana Zemskova (https://github.com/wingrune)
# ------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
import json
import random
from PIL import Image
import numpy as np
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask
from torch.nn import functional as F


class MultiStepDataset(Dataset):
    """Multistep Rollout Dataset."""

    def __init__(self, img_root, annotations_path, mode="train", transform=None):
        """
        Args:
            root_dir (string): Directory with the train and test images and annotations
            test: Flag to indicate if the train or test set is used
        """
        assert mode in ["train", "test"], "Only train and test modes supported"
        self.mode = mode
        print(self.mode)
        print(annotations_path)

        with open(annotations_path) as f:
            self.annotations = json.load(f)
        # remove trailing slash if present
        self.img_dir = img_root if img_root[-1] != "/" else img_root[:-1]
        self.transform = transform
        # interactive
        self.idx = -1
        self.actions = []

    def reset(self):

        self.idx += 1
        if self.idx >= len(self.annotations["data"]):
            self.idx = 0
        self.actions = []
        scene = self.annotations["data"][self.idx]

        state_name = scene["root"]
        state = scene["state_table"][state_name]
        actions = self.actions
        frames = []
        masks = []
        widths = []
        heights = []
        tasks = []
        initial_img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
        for i in range(len(self.actions)+1):
            # load image
            img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
            image = np.array(Image.open(img_path).resize((240, 320)))
            ann_path = "{}_mask/{}/{}.png".format(self.img_dir, scene["scene_name"], state_name)
            sem_seg_gt = np.array(Image.open(ann_path).resize((240, 320), resample=Image.NEAREST))
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
            
            if i < len(actions):
                state_name = state["actions"][actions[i]]
                state = scene["state_table"][state_name]

        sample = {
            'frames': torch.stack(frames, dim=0).unsqueeze(0),
            "masks": torch.stack(masks, dim=0).unsqueeze(0),
            "actions": torch.tensor([self.ACTIONS.index(a) for a in actions], dtype=torch.long).unsqueeze(0),
            "height": [heights],
            "width": [widths],
            "task": [tasks],
            "episode_ids": self.idx,
            "initial_image_path": [initial_img_path]
        }

        return sample

    def step(self, action):

        self.actions.append(self.ACTIONS[action])
        scene = self.annotations["data"][self.idx]

        state_name = scene["root"]
        state = scene["state_table"][state_name]
        actions = self.actions
        frames = []
        masks = []
        widths = []
        heights = []
        tasks = []
        initial_img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
        for i in range(len(self.actions)+1):
            # load image
            img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
            image = np.array(Image.open(img_path).resize((240, 320)))
            ann_path = "{}_mask/{}/{}.png".format(self.img_dir, scene["scene_name"], state_name)
            sem_seg_gt = np.array(Image.open(ann_path).resize((240, 320), resample=Image.NEAREST))
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
            
            if i < len(actions):
                #print("Len action", len(actions))
                #print(state["actions"])
                if actions[i] in state["actions"]:
                    #print(actions[i])
                    state_name = state["actions"][actions[i]]
                    state = scene["state_table"][state_name]
                    #print(state_name)
                    #exit()
                else:
                    print(i)
                    print(idx)
                    print(state_name)
                    print(scene["scene_name"])
                    print(state["actions"])
                    exit()

        sample = {
            'frames': torch.stack(frames, dim=0).unsqueeze(0),
            "masks": torch.stack(masks, dim=0).unsqueeze(0),
            "actions": torch.tensor([ACTIONS.index(a) for a in actions], dtype=torch.long).unsqueeze(0),
            "height": [heights],
            "width": [widths],
            "task": [tasks],
            "episode_ids": self.idx,
            "initial_image_path": [initial_img_path]
        }
    
        return sample
    def __len__(self):
        return len(self.annotations["data"])

    def __getitem__(self, idx, actions=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scene = self.annotations["data"][idx]

        state_name = scene["root"]
        state = scene["state_table"][state_name]

        actions = [random.choice(self.annotations["metadata"]["actions"]) for _ in range(5)]
        ACTIONS = self.annotations["metadata"]["actions"]

        frames = []
        masks = []
        widths = []
        heights = []
        tasks = []
        initial_img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)
        for i in range(5):
            # load image
            img_path = "{}/{}/{}.jpg".format(self.img_dir, scene["scene_name"], state_name)

            image = np.array(Image.open(img_path).resize((240, 320)))

            if "ai2thor" in img_path:
                ann_path = "{}_mask/{}/{}_sem.png".format(self.img_dir, scene["scene_name"], state_name)
            else:
                ann_path = "{}_mask/{}/{}.png".format(self.img_dir, scene["scene_name"], state_name)
            sem_seg_gt = np.array(Image.open(ann_path).resize((240, 320), resample=Image.NEAREST))
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
        
            if i < 4:

                if actions[i] in state["actions"]:
                    state_name = state["actions"][actions[i]]
                    state = scene["state_table"][state_name]
                else:
                    print(i)
                    print(idx)
                    print(state_name)
                    print(scene["scene_name"])
                    print(state["actions"])
                    exit()

        sample = {
            'frames': frames,
            "masks": masks,
            "actions": [ACTIONS.index(a) for a in actions],
            "height": heights,
            "width": widths,
            "task": tasks,
            "episode_ids": idx,
            "initial_image_path": initial_img_path
        }

        return sample
