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

class CrossAttention(nn.Module):
    def __init__(self, heads=1, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        query_dim=dim_head
        key_dim=dim_head
        value_dim=dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        
    def forward(self, fpsf, fimg):
        h = self.heads

        B, C, H, W = fimg.shape
        fimg = fimg.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        fpsf = fpsf.reshape(B,1,C)  # [B,1,C]
        
        q = self.to_q(fpsf) # [B, 1, C*h]
        k = self.to_k(fimg) # [B, H*W, C*h]
        
        q, k= map(lambda t: t.reshape(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k))
        
        # attention weight
        out = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # [B,h,1,H*W]
        out=out.reshape(B,C,H,W)
        return out

class PSFEstimator(nn.Module):
    def __init__(self, nf,attention):
        super().__init__()

        if attention:
            self.fusion = CrossAttention(heads=hidden_channel,dim_head=hidden_channel)
        else:
            self.fusion = nn.Conv2d(nf * 3, nf, 1, 1, 0)
        self.attention=attention
        self.body = nn.Sequential(
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            ResBlockv2(nf, nf, ksize=3, stride=1),
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            ResBlockv2(nf, nf, ksize=3, stride=1),
            ResBlockv2(nf, nf, ksize=3, stride=2, downsample=nn.AvgPool2d(3, 2, 1)),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, fimg, fpsf):
        b, c, h, w = fimg.shape

        if self.attention:
            f = self.fusion(fpsf,fimg)
        else:
            fpsf = fpsf.view(b, c, 1, 1).repeat(1, 1, h, w)
            f = torch.cat([fimg, fimg, fpsf], dim=1)
            f = self.fusion(f)

        f = self.body(f)
        return f

class Reconstructor(nn.Module):
    def __init__(
        self, hidden_channel, block_num, attention
    ):
        super().__init__()

        if attention:
            self.fusion = CrossAttention(heads=hidden_channel,dim_head=hidden_channel)
        else:
            self.fusion = nn.Conv2d(hidden_channel * 3, hidden_channel, 1, 1, 0)

        self.attention=attention
        act = partial(nn.LeakyReLU, 0.2, True)
        body = [ResidualBlockNoBN(num_feat=hidden_channel,bn=True,res_scale=1, act=act) for _ in range(block_num)]
        self.body = nn.Sequential(*body)

    def forward(self, fimg, fpsf):
        b, c, h, w = fimg.shape

        if self.attention:
            f = self.fusion(fpsf,fimg)
        else:
            fpsf = fpsf.view(b, c, 1, 1).repeat(1, 1, h, w)
            f = torch.cat([fimg, fimg, fpsf], dim=1)
            f = self.fusion(f)
        
        f = self.body(f) + f
       
        return f


class NLOSFormer(nn.Module):
    def __init__(self, input_channel, hidden_channel=64, psf_dim=42,block_num=16,attention=False):
        super().__init__()

        self.psf_dim = psf_dim

        self.img_head = nn.Conv2d(input_channel, hidden_channel, 3, 1, 1)
        self.psf_head = nn.Conv2d(psf_dim, hidden_channel, 1, 1, 0)

        self.PSFEstimator = PSFEstimator(hidden_channel,attention)
        self.Reconstructor = Reconstructor(hidden_channel, block_num,attention)
        
        self.img_tail = nn.Conv2d(hidden_channel, input_channel, 3, 1, 1)
        self.psf_tail = nn.Conv2d(hidden_channel, psf_dim, 1, 1, 0)

        # Set initial kernel to zero
        self.init_kernel = nn.Parameter(torch.zeros(1, psf_dim, 1, 1), requires_grad=True)

    def loadkernel(self,matfile):
        reduced_kernel=loadmat(matfile)["reduced_kernel"]
        self.setkernel(reduced_kernel)

    def setkernel(self,reduced_ker):
        self.init_kernel = nn.Parameter(torch.from_numpy(reduced_ker).view(1, self.kernel_dim, 1, 1).float(), requires_grad=True)

    def forward(self, img):
        b, c, h, w = img.shape

        fimg = self.img_head(img) # [b,hidden_channel,h,w]
        fpsf = self.psf_head(self.init_kernel).repeat(b, 1, 1, 1) # [b,hidden_channel,1,1]
        
        fpsf = self.PSFEstimator(fimg, fpsf) + fpsf # the residual connection is important
        fimg = self.Reconstructor(fimg, fpsf) + fimg 
        
        result = self.img_tail(fimg) # + self.mean # (b, c,h,w)
        psf = self.psf_tail(fpsf).view(b, -1) # (b, kernel_num)

        return result, psf