import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset
from datetime import datetime

class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "CCViMUNet"
        model_mode = "fast" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = None # number of nuclear types (including background) kumar is None

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = False

        # shape information -
        # below config is for original mode.
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [256, 256] # patch shape at output of network

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [256,256]:
                raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dataset_name = "cpm17"  # extracts dataset info from dataset_CKC.py
        self.log_dir = f"/kumar/output/CCM_UNET_{current_time}" # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [
            "/kumar/processed/kumar/train/540x540_164x164"
        ]
        self.valid_dir_list = [
            "/kumar/processed/kumar/valid/540x540_164x164"
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
