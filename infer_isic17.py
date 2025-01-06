import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.CCViMUNet.CCViMUNet import LCVMUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting_isic17 import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'CCViM_isic17':
        model = LCVMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model_weight = torch.load("./pre_trained_weights/isic17.pth")
        model.load_state_dict(model_weight, strict=False)
    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------validate----------#')

    torch.cuda.empty_cache()

    loss, miou = val_one_epoch(
        val_loader,
        model,
        criterion,
        0,
        logger,
        config
    )


if __name__ == '__main__':
    config = setting_config
    main(config)