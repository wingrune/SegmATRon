from utils.config_utils import (
    get_config,
    get_args,
    build_model,
    build_evaluator
)
import torch
import random
import numpy
import json

import wandb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)



def evaluate(seed, cfg):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    model = build_model(cfg)
    evaluator = build_evaluator(model, cfg, load_checkpoint=True)
    evaluator.evaluate(save_results=True)

if __name__ == "__main__":
    args = get_args()
    cfg = get_config(args.config_file)
    evaluate(cfg.RANDOM_SEED, cfg)