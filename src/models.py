import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class ENet(nn.Module):
    def __init__(self, output_size=1, model_name='efficientnet-b0'):
        super(ENet, self).__init__()
        self.enet = EfficientNet.from_pretrained(model_name=model_name, num_classes=output_size)

    def forward(self, x):
        out = self.enet(x)

        return out
