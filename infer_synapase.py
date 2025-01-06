import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.dataset import RandomGenerator
from engine_synapse import *

from models.CCViMUNet.CCViMUNet import LCVMUNet

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting_synapse import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()
    gpus_type, gpus_num = torch.cuda.get_device_name(), torch.cuda.device_count()
    if config.distributed:
        print('#----------Start DDP----------#')
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(config.seed)
        config.local_rank = torch.distributed.get_rank()

    print('#----------Preparing dataset----------#')
    train_dataset = config.datasets(base_dir=config.data_path, list_dir=config.list_dir, split="train",
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[config.input_size_h, config.input_size_w])]))
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size // gpus_num if config.distributed else config.batch_size,
                              shuffle=(train_sampler is None),
                              pin_memory=True,
                              num_workers=config.num_workers,
                              sampler=train_sampler)

    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
    val_loader = DataLoader(val_dataset,
                            batch_size=1,  # if config.distributed else config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            sampler=val_sampler,
                            drop_last=True)

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    if config.network == 'CCViM_synapse':
        model = LCVMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model_weight = torch.load("./pre_trained_weights/best_synapse.pth")
        model.load_state_dict(model_weight, strict=False)
    else:
        raise ('Please prepare a right net!')

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    mean_dice, mean_hd95 = val_one_epoch(
        val_dataset,
        val_loader,
        model,
        0,
        logger,
        config,
        test_save_path=outputs,
        val_or_test=False
        )



if __name__ == '__main__':
    config = setting_config
    main(config)