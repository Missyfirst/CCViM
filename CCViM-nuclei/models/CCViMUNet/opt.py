import torch.optim as optim

from run_utils.callbacks.base import (
    AccumulateRawOutput,
    PeriodicSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    VisualizeOutput,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events

from models.targets import gen_targets, prep_sample
from models.CCViMUNet.CCViMUNet import LCVMUNet as create_model
from models.run_desc import proc_valid_step_output, train_step, valid_step, viz_step_output


# TODO: training config only ?
# TODO: switch all to function name String for all option
def get_config(nr_type, mode):
    return {
        # ------------------------------------------------------------------
        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        "phase_list": [
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: create_model(
                            input_channels=3, nr_types=nr_type,freeze=True,
                            load_ckpt_path="./pre_trained_weights/local_vssm_small.ckpt",
                        ),
                        "optimizer": [
                            optim.AdamW,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-3,  # initial learning rate,
                                "betas": (0.9, 0.999),
                                "eps": 1e-8,
                                "weight_decay": 1e-2,
                            },
                        ],
                        "lr_scheduler": lambda x: optim.lr_scheduler.CosineAnnealingLR(x, T_max=100, eta_min=0.00001),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        'pretrained': 1,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 32, "valid": 32,},  # engine name : value
                "nr_epochs": 100,
            },
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: create_model(
                            input_channels=3, nr_types=nr_type,freeze=False,
                            load_ckpt_path=None,
                        ),
                        "optimizer": [
                            optim.AdamW,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-3,  # initial learning rate,
                                "betas": (0.9, 0.999),
                                "eps": 1e-8,
                                "weight_decay": 1e-2,
                            },
                        ],
                        "lr_scheduler": lambda x: optim.lr_scheduler.CosineAnnealingLR(x, T_max=100, eta_min=0.00001),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        "pretrained": -1,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 32, "valid": 32,},  # batch size per gpu
                "nr_epochs": 200,
            },
        ],
        # ------------------------------------------------------------------
        # TODO: dynamically for dataset plugin selection and processing also?
        # all enclosed engine shares the same neural networks
        # as the on at the outer calling it
        "run_engine": {
            "train": {
                # TODO: align here, file path or what? what about CV?
                "dataset": "",  # whats about compound dataset ?
                "nr_procs": 16,  # number of threads for dataloader
                "run_step": train_step,  # TODO: function name or function variable ?
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(), # TODO: very slow, may be due to back forth of tensor/numpy ?
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(), # save checkpoints
                        VisualizeOutput(viz_step_output), # 把输出的图像进行可视化
                        LoggingEpochOutput(),
                        TriggerEngine("valid"), # 运行验证的
                        ScheduleLr(),
                    ],
                },
            },
            "valid": {
                "dataset": "",  # whats about compound dataset ?
                "nr_procs": 8,  # number of threads for dataloader
                "run_step": valid_step,
                "reset_per_run": True,  # * to stop aggregating output etc. from last run
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        # TODO: is there way to preload these ?
                        ProcessAccumulatedRawOutput(
                            lambda a: proc_valid_step_output(a, nr_types=nr_type)
                        ),
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }
