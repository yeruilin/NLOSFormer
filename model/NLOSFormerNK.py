import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from scipy.io import loadmat

def get_activation(activation, activation_params=None, num_channels=None):
    if activation_params is None:
        activation_params = {}

    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'lrelu':
        return nn.LeakyReLU(negative_slope=activation_params.get('negative_slope', 0.1), inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=num_channels)
    elif activation == 'none':
        return None
    else:
        raise Exception('Unknown activation {}'.format(activation))


def get_attention(attention_type, num_channels=None):
    if attention_type == 'none':
        return None
    else:
        raise Exception('Unknown attention {}'.format(attention_type))


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=False, activation='relu', padding_mode='zeros', activation_params=None):
    layers = []

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))

    activation_layer = get_activation(activation, activation_params, num_channels=out_planes)
    if activation_layer is not None:
        layers.append(activation_layer)

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm=False, activation='relu',
                 padding_mode='zeros', attention='none'):
        super(ResBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, kernel_size=3, padding=1, stride=stride, dilation=dilation,
                                batch_norm=batch_norm, activation=activation, padding_mode=padding_mode)

        self.conv2 = conv_block(planes, planes, kernel_size=3, padding=1, dilation=dilation, batch_norm=batch_norm,
                                activation='none', padding_mode=padding_mode)

        self.downsample = downsample
        self.stride = stride

        self.activation = get_activation(activation, num_channels=planes)
        self.attention = get_attention(attention_type=attention, num_channels=planes)

    def forward(self, x):
        residual = x

        out = self.conv2(self.conv1(x))

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention is not None:
            out = self.attention(out)

        out += residual

        out = self.activation(out)

        return out

class ResBlockv2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ksize=3, stride=1, downsample=None, dilation=1, norm=None, activation='relu',
                 padding_mode='zeros', attention='none'):
        super().__init__()

        self.body = nn.Sequential(
            norm(inplanes) if norm else nn.Identity(),
            get_activation(activation),
            nn.Conv2d(inplanes, planes, ksize, stride, ksize//2),
            norm(inplanes) if norm else nn.Identity(),
            get_activation(activation),
            nn.Conv2d(planes, planes, ksize, 1, ksize//2)
            )

        self.downsample = downsample
        self.stride = stride

        self.attention = get_attention(attention_type=attention, num_channels=planes)

    def forward(self, x):
        residual = x

        out = self.body(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention is not None:
            out = self.attention(out)

        out += residual

        return out

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    from https://github.com/greatlog/DAN.git
    """

    def __init__(self, num_feat=64, bn=True, res_scale=1, pytorch_init=False, act=partial(nn.ReLU, inplace=True)):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = act()
        if bn:
            self.bn1=nn.BatchNorm2d(num_feat)
            self.bn2=nn.BatchNorm2d(num_feat)
        else:
            self.bn1=nn.Identity()
            self.bn2=nn.Identity()

    def forward(self, x):
        identity = x
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return identity + out * self.res_scale


# NLOSFormer No Kernel (NLOSFormerNK)
class NLOSFormerNK(nn.Module):
    def __init__(self, input_channel, hidden_channel=64, block_num=16):
        super().__init__()

        self.img_head = nn.Conv2d(input_channel, hidden_channel, 3, 1, 1)
        
        act = partial(nn.LeakyReLU, 0.2, True)
        body = [ResidualBlockNoBN(num_feat=hidden_channel,bn=True,res_scale=1, act=act) for _ in range(block_num)]
        self.body = nn.Sequential(*body)
        
        self.img_tail = nn.Conv2d(hidden_channel, input_channel, 3, 1, 1)

    def forward(self, img):
        b, c, h, w = img.shape

        fimg = self.img_head(img) # [b,c,h,w]
        fimg = self.body(fimg)
        result = self.img_tail(fimg) # (b, c,h,w)

        return result