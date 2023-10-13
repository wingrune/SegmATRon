# ------------------------------------------------------------------------------
# Reference: https://github.com/allenai/interactron/blob/main/evaluate.py
# Modified by Tatiana Zemskova (https://github.com/wingrune)
# ------------------------------------------------------------------------------

from utils.config_utils import (
    get_config,
    get_args,
    build_model,
    build_evaluator
)
import torch
import random
import numpy

import wandb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)



def evaluate():
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    args = get_args()
    cfg = get_config(args.config_file)
    model = build_model(cfg)
    evaluator = build_evaluator(model, cfg, load_checkpoint=True)
    evaluator.evaluate(save_results=True)


if __name__ == "__main__":
    evaluate()
