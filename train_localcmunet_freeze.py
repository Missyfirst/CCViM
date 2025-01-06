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
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
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
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'lovalcmunet_freeze03_fea16_batch28':
        model = LCVMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
            freeze=True,
        )
        # model.load_from()
        model_weight = torch.load("/opt/data/private/zhuyun/MedImage/VM-UNet-main/pre_trained_weights/best-epoch133-loss0.2397.pth")
        model.load_state_dict(model_weight, strict=False)
        load_feedback = model.load_state_dict(model_weight, strict=False)
        # # * uncomment for your convenience
        # print("Missing Variables: \n", load_feedback[0])
        # print("Detected Unknown Variables: \n", load_feedback[1])
    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    max_miou = 0

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss, max_miou = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss'], checkpoint['max_miou']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}, max_miou:{max_miou:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, 101):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss, miou = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        if miou > max_miou:
        # if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            max_miou = miou
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'max_miou': max_miou,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    ###########################################################################
    print('#----------Prepareing Model unfreezing----------#')
    model = LCVMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
        freeze=False,
    )
    modelCheckpoint = torch.load(os.path.join(checkpoint_dir, 'latest.pth'))
    pretrained_dict = modelCheckpoint['model_state_dict']
    model.load_state_dict(pretrained_dict, strict=False)
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, 201):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss, miou = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        if miou > max_miou:
            # if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            max_miou = miou
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'max_miou': max_miou,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))




    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)