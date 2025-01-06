import torch
from torch import nn
from models.CCViMUNet.CCViM import CCViM
 ## localmamba and ccmamba

class LCVMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 2, 2],
                 depths_decoder=[2, 2, 2, 1],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.lcvmunet = CCViM(input_channels=input_channels,num_classes=num_classes,depths=depths,
                                      decoder_depth=depths_decoder,
                                      drop_path_rate=drop_path_rate)


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.lcvmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits


    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.lcvmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)

            if 'state_dict' in modelCheckpoint:
                # 如果存在 'state_dict' 键
                print(f'#####')
                pretrained_dict = modelCheckpoint['state_dict']
            if 'scheduler_state_dict' in modelCheckpoint:
                # 如果存在 'scheduler_state_dict' 键
                print(f'#####')
                pretrained_dict = modelCheckpoint['scheduler_state_dict']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.lcvmunet.load_state_dict(model_dict)  # 加载预训练参数

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.lcvmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            if 'state_dict' in modelCheckpoint:
                # 如果存在 'state_dict' 键
                print(f'#####')
                pretrained_odict = modelCheckpoint['state_dict']
            if 'scheduler_state_dict' in modelCheckpoint:
                # 如果存在 'scheduler_state_dict' 键
                print(f'#####')
                pretrained_odict = modelCheckpoint['scheduler_state_dict']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.lcvmunet.load_state_dict(model_dict)

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")

if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile

    model = LCVMUNet(load_ckpt_path="/opt/data/private/zhuyun/MedImage/VM-UNet-main/pre_trained_weights/local_vssm_small.ckpt",)


    model.load_from()
