# ------------------------------------------------------------------------------
# Reference: https://github.com/SHI-Labs/OneFormer/blob/main/oneformer/__init__.py
# Modified by Tatiana Zemskova (https://github.com/wingrune)
# ------------------------------------------------------------------------------
from . import data  # register all new datasets
from . import modeling

# config
from .config import *

# dataset loading
from .data.dataset_mappers.coco_unified_new_baseline_dataset_mapper import COCOUnifiedNewBaselineDatasetMapper
from .data.dataset_mappers.oneformer_unified_dataset_mapper import (
    OneFormerUnifiedDatasetMapper,
)

# models
from .oneformer_model import OneFormer

