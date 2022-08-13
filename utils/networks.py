import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from utils import experiment_manager
from copy import deepcopy
from collections import OrderedDict
from sys import stderr


def save_checkpoint(network, optimizer, epoch, step, cfg: experiment_manager.CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch, cfg: experiment_manager.CfgNode, device):
    net = DualStreamPopulationNet(cfg.MODEL) if cfg.MODEL.DUALSTREAM else PopulationNet(cfg.MODEL)
    net.to(device)

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']


def create_ema_network(net, cfg):
    ema_net = EMA(net, decay=cfg.CONSISTENCY_TRAINER.WEIGHT_DECAY)
    return ema_net


class DualStreamPopulationNet(nn.Module):

    def __init__(self, dual_model_cfg: experiment_manager.CfgNode):
        super(DualStreamPopulationNet, self).__init__()
        self.dual_model_cfg = dual_model_cfg
        self.stream1_cfg = dual_model_cfg.STREAM1
        self.stream2_cfg = dual_model_cfg.STREAM2

        self.stream1 = PopulationNet(self.stream1_cfg, enable_fc=False)
        self.stream2 = PopulationNet(self.stream2_cfg, enable_fc=False)

        stream1_num_ftrs = self.stream1.model.fc.in_features
        stream2_num_ftrs = self.stream2.model.fc.in_features
        self.outc = nn.Linear(stream1_num_ftrs + stream2_num_ftrs, dual_model_cfg.OUT_CHANNELS)
        self.relu = torch.nn.ReLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        features1 = self.stream1(x1)
        features2 = self.stream2(x2)
        p1 = self.relu(self.stream1.model.fc(features1))
        p2 = self.relu(self.stream2.model.fc(features2))
        features_fusion = torch.cat((features1, features2), dim=1)
        p_fusion = self.relu(self.outc(features_fusion))
        return p_fusion, p1, p2


class PopulationNet(nn.Module):

    def __init__(self, model_cfg, enable_fc: bool = True):
        super(PopulationNet, self).__init__()
        self.model_cfg = model_cfg
        self.enable_fc = enable_fc
        pt = model_cfg.PRETRAINED
        assert (model_cfg.TYPE == 'resnet')
        if model_cfg.SIZE == 18:
            self.model = torchvision.models.resnet18(pretrained=pt)
        elif model_cfg.SIZE == 50:
            self.model = torchvision.models.resnet50(pretrained=pt)
        else:
            raise Exception(f'Unkown resnet size ({model_cfg.SIZE}).')

        new_in_channels = model_cfg.IN_CHANNELS

        if new_in_channels != 3:
            # only implemented for resnet
            assert (model_cfg.TYPE == 'resnet')

            first_layer = self.model.conv1
            # Creating new Conv2d layer
            new_first_layer = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=first_layer.bias
            )
            # he initialization
            nn.init.kaiming_uniform_(new_first_layer.weight.data, mode='fan_in', nonlinearity='relu')
            if new_in_channels > 3:
                # replace weights of first 3 channels with resnet rgb ones
                first_layer_weights = first_layer.weight.data.clone()
                new_first_layer.weight.data[:, :first_layer.in_channels, :, :] = first_layer_weights
            # if it is less than 3 channels we use he initialization (no pretraining)

            # replacing first layer
            self.model.conv1 = new_first_layer
            # https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379/2
            # https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10

        # replacing fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, model_cfg.OUT_CHANNELS)
        self.relu = torch.nn.ReLU()
        self.encoder = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_fc:
            x = self.model(x)
            x = self.relu(x)
        else:
            x = self.encoder(x)
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x


# https://www.zijianhu.com/post/pytorch/ema/
class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.ema_model = deepcopy(self.model)

        for param in self.ema_model.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        ema_model_params = OrderedDict(self.ema_model.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == ema_model_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            ema_model_params[name].sub_((1. - self.decay) * (ema_model_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        ema_model_buffers = OrderedDict(self.ema_model.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == ema_model_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            ema_model_buffers[name].copy_(buffer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.ema_model(inputs)

    def get_ema_model(self):
        return self.ema_model

    def get_model(self):
        return self.model


if __name__ == '__main__':
    x = torch.randn(1, 5, 224, 224)
    model = torchvision.models.vgg16(pretrained=False)  # pretrained=False just for debug reasons
    first_conv_layer = [nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features = nn.Sequential(*first_conv_layer)
    output = model(x)