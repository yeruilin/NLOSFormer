import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum

from model.Uformer import Upsample,Downsample,InputProj,OutputProj,BasicUformerLayer

import warnings
warnings.filterwarnings("ignore", 
    message="Grad strides do not match bucket view strides.*")

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

####################asymmetric cross-fusion module###################
class ACFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ACFM, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

#################################################
class SCM(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_plane, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-in_plane, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

###############Transfer function###############
class Exchange1(nn.Module):
    def __init__(self):
        super(Exchange1, self).__init__()

    def forward(self, x):
        bs, c, h, w = x.size()

        # flaten
        exc = rearrange(x, ' b c h w -> b (h w) c', h=h, w=w)
        return exc


class Exchange2(nn.Module):
    def __init__(self):
        super(Exchange2, self).__init__()

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        # spatial restore
        exc = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        return exc


#########################################################################
###########lightweight multi-scale NLOS network##############
class LMSNet(nn.Module):
    def __init__(self,num_res=8, img_size=256, in_chans=1, dd_in=1,
                 embed_dim=32, depths=[1, 1, 1, 1, 1, 1, 1, 1, 1], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='pefns', token_attn='SCIA',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True, modulator=False,
                 cross_modulator=False, **kwargs):
        super(LMSNet, self).__init__()

        base_channel = 32
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        # self.token_attn = token_attn
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Exchange
        self.exchange1 = Exchange1()
        self.exchange2 = Exchange2()

        # Input/Output
        # self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1,
        #                             act_layer=nn.LeakyReLU)
        # self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        # self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
        #                                         output_dim=embed_dim * 8,
        #                                         input_resolution=(img_size // (2 ** 3),
        #                                                           img_size // (2 ** 3)),
        #                                         depth=depths[3],
        #                                         num_heads=num_heads[3],
        #                                         win_size=win_size,
        #                                         mlp_ratio=self.mlp_ratio,
        #                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                         drop=drop_rate, attn_drop=attn_drop_rate,
        #                                         drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
        #                                         norm_layer=norm_layer,
        #                                         use_checkpoint=use_checkpoint,
        #                                         token_projection=token_projection, token_mlp=token_mlp,
        #                                         token_attn=token_attn, shift_flag=shift_flag)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)


        self.feat_extract = nn.ModuleList([
            BasicConv(in_chans, base_channel, kernel_size=3, relu=True, stride=1),
            # BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            # BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            # BasicConv(base_channel*8, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            # BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, in_chans, kernel_size=3, relu=False, stride=1)
        ])

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),
                                                        img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      token_projection=token_projection, token_mlp=token_mlp, token_attn=token_attn, shift_flag=shift_flag)


        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        # self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
        #                                         output_dim=embed_dim * 16,
        #                                         input_resolution=(img_size // (2 ** 3),
        #                                                           img_size // (2 ** 3)),
        #                                         depth=depths[5],
        #                                         num_heads=num_heads[5],
        #                                         win_size=win_size,
        #                                         mlp_ratio=self.mlp_ratio,
        #                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                         drop=drop_rate, attn_drop=attn_drop_rate,
        #                                         drop_path=dec_dpr[:depths[5]],
        #                                         norm_layer=norm_layer,
        #                                         use_checkpoint=use_checkpoint,
        #                                         token_projection=token_projection, token_mlp=token_mlp,
        #                                         token_attn=token_attn, shift_flag=shift_flag,
        #                                         modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_1 = upsample(embed_dim * 8, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_2 = upsample(embed_dim * 4, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_3 = upsample(embed_dim * 2, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)


        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, in_chans, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, in_chans, kernel_size=3, relu=False, stride=1),
            ]
        )
        # asymmetric cross-fusion module (ACFM)
        self.AFFs = nn.ModuleList([
            ACFM(base_channel * 7, base_channel*1),
            ACFM(base_channel * 7, base_channel*2),
            ACFM(base_channel * 7, base_channel * 4)
        ])
        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(in_chans,base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(in_chans,base_channel * 2)
        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)
        self.drop3 = nn.Dropout2d(0.1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None):
        # x is [b,1,h,w]
        x_2 = F.interpolate(x, scale_factor=0.5) # [b,1,h//2,w//2]
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        #Encoder

        y = self.feat_extract[0](x)
        y = self.exchange1(y)

        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)

        pool0 = self.exchange2(pool0)
        y=pool0
        # 第2层
        y = self.FAM2(y, z2)

        y = self.exchange1(y)
        conv1 = self.encoderlayer_1(y, mask=mask)
        pool1 = self.dowsample_1(conv1)

        conv0 = self.exchange2(conv0)
        conv1 = self.exchange2(conv1)
        pool1 = self.exchange2(pool1)
        z=pool1
        z = self.FAM1(z, z4)
        z = self.exchange1(z)

        conv2 = self.encoderlayer_2(z, mask=mask)
        pool2 = self.dowsample_2(conv2)

        pool3 = self.dowsample_3(pool2)
        # Bottleneck
        conv3 = self.conv(pool3, mask=mask)

        conv2 = self.exchange2(conv2)
        z12 = F.interpolate(conv0, scale_factor=0.5)
        z21 = F.interpolate(conv1, scale_factor=2)
        z42 = F.interpolate(conv2, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        z23 = F.interpolate(conv1, scale_factor=0.5)
        z13 = F.interpolate(conv0, scale_factor=0.25)

        s3 = self.AFFs[2](z13, z23, conv2)
        s2 = self.AFFs[1](z12, conv1, z42)
        s1 = self.AFFs[0](conv0, z21, z41)

        s3 = self.drop3(s3)
        s2 = self.drop2(s2)
        s1 = self.drop1(s1)

        # Decoder
        up0 = self.upsample_0(conv3)
        up0 = self.upsample_1(up0)
        z = up0
        z = self.exchange2(z)
        z = torch.cat([z, s3], dim=1)
        z = self.Convs[0](z)
        z = self.exchange1(z)
        z = self.decoderlayer_1(z, mask=mask)
        z = self.exchange2(z)
        z_ = self.ConvsOut[0](z)
        outputs.append(z_+x_4)
        z = self.exchange1(z)
        up1 = self.upsample_2(z)
        z = up1
        z = self.exchange2(z)
        z = torch.cat([z, s2], dim=1)
        z = self.Convs[1](z)
        z = self.exchange1(z)
        z = self.decoderlayer_2(z, mask=mask)
        z = self.exchange2(z)
        z_ = self.ConvsOut[1](z)
        outputs.append(z_+x_2)
        z = self.exchange1(z)
        up2 = self.upsample_3(z)
        z = up2

        z = self.exchange2(z)
        z = torch.cat([z, s1], dim=1)
        z = self.Convs[2](z)
        z = self.exchange1(z)

        z = self.decoderlayer_3(z, mask=mask)
        z = self.exchange2(z)
        z = self.feat_extract[1](z)
        outputs.append(z+x)
        
        return outputs
        # return z+x

    
###########################################################################
####################multi-scale NLOS imaging network framework#############
class MSNet(nn.Module):
    def __init__(self,num_res=8, img_size=256, in_chans=1, dd_in=1,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='pefn', token_attn='SCIA',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True, modulator=False,
                 cross_modulator=False, **kwargs):
        super(MSNet, self).__init__()

        base_channel = 32
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]


        # Exchange
        self.exchange1 = Exchange1()
        self.exchange2 = Exchange2()

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        #
        self.feat_extract = nn.ModuleList([
            BasicConv(in_chans, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*8, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, in_chans, kernel_size=3, relu=False, stride=1)
        ])

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),
                                                        img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      token_projection=token_projection, token_mlp=token_mlp, token_attn=token_attn, shift_flag=shift_flag)


        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_1 = upsample(embed_dim * 8, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_2 = upsample(embed_dim * 4, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_3 = upsample(embed_dim * 2, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                token_attn=token_attn, shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)

        #
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)
        ])
        #
        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, in_chans, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, in_chans, kernel_size=3, relu=False, stride=1),
            ]
        )
        # asymmetric cross-fusion module (ACFM)
        self.AFFs = nn.ModuleList([
            ACFM(base_channel * 7, base_channel*1),
            ACFM(base_channel * 7, base_channel*2),
            ACFM(base_channel * 7, base_channel * 4)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)
        self.drop3 = nn.Dropout2d(0.1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None):

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        #Encoder

        y = self.feat_extract[0](x)
        # Exchange1(bs,hw,c)
        y = self.exchange1(y)

        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)

        # Exchange2（bs,c,h,w）
        pool0 = self.exchange2(pool0)
        y=pool0
        # 第2层
        y = self.FAM2(y, z2)

        # Exchange1(bs,hw,c)
        y = self.exchange1(y)
        conv1 = self.encoderlayer_1(y, mask=mask)
        pool1 = self.dowsample_1(conv1)

        # Exchange2（bs,c,h,w）
        conv0 = self.exchange2(conv0)
        conv1 = self.exchange2(conv1)
        # Exchange2（bs,c,h,w）
        pool1 = self.exchange2(pool1)
        z=pool1

        # z = self.feat_extract[2](pool1)
        z = self.FAM1(z, z4)
        # Exchange1(bs,hw,c)
        z = self.exchange1(z)

        conv2 = self.encoderlayer_2(z, mask=mask)
        pool2 = self.dowsample_2(conv2)

        pool3 = self.dowsample_3(pool2)
        # Bottleneck
        conv3 = self.conv(pool3, mask=mask)

        # Exchange2（bs,c,h,w）
        conv2 = self.exchange2(conv2)
        z12 = F.interpolate(conv0, scale_factor=0.5)
        z21 = F.interpolate(conv1, scale_factor=2)
        z42 = F.interpolate(conv2, scale_factor=2)   ##############这里修改了##########
        z41 = F.interpolate(z42, scale_factor=2)
        z23 = F.interpolate(conv1, scale_factor=0.5)
        z13 = F.interpolate(conv0, scale_factor=0.25)

        s3 = self.AFFs[2](z13, z23, conv2)
        s2 = self.AFFs[1](z12, conv1, z42)
        s1 = self.AFFs[0](conv0, z21, z41)

        s3 = self.drop3(s3)
        s2 = self.drop2(s2)
        s1 = self.drop1(s1)

        # Decoder

        up0 = self.upsample_0(conv3)
        up0 = self.upsample_1(up0)
        z = up0
        z = self.exchange2(z)
        z = torch.cat([z, s3], dim=1)
        z = self.Convs[0](z)
        z = self.exchange1(z)
        z = self.decoderlayer_1(z, mask=mask)
        z = self.exchange2(z)
        z_ = self.ConvsOut[0](z)
        outputs.append(z_+x_4)
        z = self.exchange1(z)

        up1 = self.upsample_2(z)
        z = up1
        z = self.exchange2(z)
        z = torch.cat([z, s2], dim=1)
        z = self.Convs[1](z)
        z = self.exchange1(z)
        z = self.decoderlayer_2(z, mask=mask)
        z = self.exchange2(z)
        z_ = self.ConvsOut[1](z)
        outputs.append(z_+x_2)

        z = self.exchange1(z)
        up2 = self.upsample_3(z)
        z = up2
        z = self.exchange2(z)
        z = torch.cat([z, s1], dim=1)
        z = self.Convs[2](z)
        z = self.exchange1(z)
        z = self.decoderlayer_3(z, mask=mask)
        z = self.exchange2(z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MS-NLOS":
        return MSNet()
    elif model_name == "LMS-NLOS":
        return LMSNet()
    raise ModelError('Wrong Model!\nYou should choose MS-NLOS or LMS-NLOS.')
