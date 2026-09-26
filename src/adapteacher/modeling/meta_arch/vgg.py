# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn
import copy
import torch
from typing import Union, List, Dict, Any, cast
from detectron2.modeling.backbone import (
    ResNet,
    Backbone,
    build_resnet_backbone,
    BACKBONE_REGISTRY
)
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, LastLevelP6P7


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

import torch.nn.functional as F
import math

class RelBias(nn.Module):
    def __init__(self, num_heads, txy_size, tz_bins):
        super().__init__()
        self.num_heads = num_heads
        self.txy_size = txy_size    # should equal win
        self.tz_bins = tz_bins      # len(bins)+1 from bucketize
        self.bias = nn.Parameter(torch.zeros(num_heads, txy_size, txy_size, tz_bins))
        nn.init.zeros_(self.bias)

    def forward(self, dx, dy, dz_bucket):
        # dx, dy in [-R..R], mapped to [0..table_size-1]
        t = self.table_size
        idx_x = dx + (t//2)
        idx_y = dy + (t//2)
        idx_z = dz_bucket
        return self.bias[:, idx_x, idx_y, idx_z]  # [H, Ww, Ww] after broadcasting

def bucketize_dz(dz, bins=(-0.2,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.2)):
    # dz: [B, HW, Ww] (float)
    device = dz.device
    edges = torch.tensor(bins, device=device).view(1,1,1,-1)
    # for each dz, count how many edges it's >= -> integer bucket in [0..len(bins)]
    idx = (dz.unsqueeze(-1) >= edges).sum(dim=-1)   # [B, HW, Ww]
    # clamp for safety (Tz = len(bins)+1)
    Tz = len(bins) + 1
    return idx.clamp_(0, Tz-1), Tz

class WindowedDepthXAttn(nn.Module):
    """
    Local cross-attn with 3D relative bias on (dx,dy,dz).
    Use on later stages (e.g., 1/8–1/16).
    """
    def __init__(self, embed_dim, num_heads=4, win=7):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.dim, self.h, self.dh, self.win = embed_dim, num_heads, embed_dim // num_heads, win
        self.q = nn.Conv2d(embed_dim, embed_dim, 1)
        self.k = nn.Conv2d(embed_dim, embed_dim, 1)
        self.v = nn.Conv2d(embed_dim, embed_dim, 1)
        self.out = nn.Conv2d(embed_dim, embed_dim, 1)
        self.depth_enc = nn.Sequential(
            nn.Conv2d(1, embed_dim//2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim, 3, padding=1)
        )
        BINS = (-0.2,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.2)
        TZ = len(BINS) + 1
        self.rel = RelBias(num_heads, txy_size=self.win, tz_bins=TZ)

    def _unfold_win(self, t):
        # [B,C,H,W] -> [B, C, H*W, win*win] keyed windows (unfold over neighborhoods)
        B, C, H, W = t.shape
        pad = self.win//2
        tpad = F.pad(t, (pad,pad,pad,pad), mode='reflect')
        patches = F.unfold(tpad, kernel_size=self.win, padding=0, stride=1)  # [B, C*win*win, H*W]
        patches = patches.view(B, C, self.win*self.win, H*W)                 # [B,C,Ww,H*W]
        return patches.permute(0,1,3,2).contiguous()                         # [B,C,H*W,Ww]

    def forward(self, x, d):
        B, C, H, W = x.shape
        HW = H * W
        r = self.win // 2
        Ww = self.win * self.win

        # depth normalize & inverse depth
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        d = (d - d.amin((2,3), True)) / (d.amax((2,3), True) - d.amin((2,3), True) + 1e-8)
        invz = 1.0 / (d + 1e-6)  # emphasize near structures

        # project features
        z = self.depth_enc(d)
        q = self.q(x)  # [B,C,H,W]
        k = self.k(z)
        v = self.v(z)

        # shape to heads
        q = q.view(B, self.h, self.dh, HW).transpose(2, 3)              # [B,h,HW,dh]

        # unfold k,v windows (centered window per location)
        k_win = self._unfold_win(k).view(B, self.h, self.dh, HW, -1)     # [B,h,dh,HW,Ww]
        v_win = self._unfold_win(v).view(B, self.h, self.dh, HW, -1)

        # align dims to multiply over dh
        k_win = k_win.permute(0, 1, 3, 2, 4)  # -> [B,h,HW,dh,Ww]
        v_win = v_win.permute(0, 1, 3, 2, 4)  # -> [B,h,HW,dh,Ww]

        # dot-product logits over dh  --> [B,h,HW,Ww]
        logits = (q.unsqueeze(-1) * k_win).sum(dim=3) / math.sqrt(self.dh)

        # ---------- 3D RELATIVE BIAS (dx, dy, dz) ----------
        # precompute window offsets (row-major order to match unfold)
        yy, xx = torch.meshgrid(
            torch.arange(-r, r+1, device=x.device),
            torch.arange(-r, r+1, device=x.device),
            indexing='ij'
        )  # [win, win]
        dx = xx.reshape(-1)  # [Ww]  note: dx = column offset
        dy = yy.reshape(-1)  # [Ww]  note: dy = row offset

        # windowed inverse-depth neighbors for each center
        # _unfold_win(invz) returns [B, 1, HW, Ww] for a 1-channel map
        invz_win = self._unfold_win(invz).squeeze(1)          # [B, HW, Ww]
        invz_ctr = invz.view(B, 1, HW).transpose(1, 2)        # [B, HW, 1]
        dz = invz_win - invz_ctr                               # [B, HW, Ww]
        dz_bucket, Tz = bucketize_dz(dz)                                # [B, HW, Ww], int
        Txy = self.win                       # matches RelBias.txy_size
        assert Tz == self.rel.tz_bins

        bias_table = self.rel.bias           # [h, Txy, Txy, Tz]
        idx_xy = (dy + (Txy//2)) * Txy + (dx + (Txy//2))   # [Ww], on the same device
        bias_lut = bias_table.view(self.h, Txy*Txy, Tz)[:, idx_xy, :]   # [h, Ww, Tz]

        # gather bias for each (B,HW,Ww) using dz_bucket
        # expand to broadcast: [1,h,1,Ww,Tz] and index with [B,1,HW,Ww,1]
        bias_lut = bias_lut.unsqueeze(0).unsqueeze(2)                 # [1,h,1,Ww,Tz]
        idx = dz_bucket.unsqueeze(1).unsqueeze(-1)                    # [B,1,HW,Ww,1]
        bias = torch.gather(bias_lut.expand(B, -1, HW, -1, -1), -1, idx).squeeze(-1)  # [B,h,HW,Ww]

        # add to logits
        logits = logits + bias

        # attention and aggregate
        attn = logits.softmax(dim=-1)                                 # [B,h,HW,Ww]
        out = (v_win * attn.unsqueeze(3)).sum(dim=-1)                 # [B,h,HW,dh]
        out = out.permute(0,1,3,2).contiguous().view(B, C, H, W)      # [B,C,H,W]
        out = self.out(out)
        return x + out

class vgg_backbone(Backbone):
    """
    Backbone (bottom-up) for FBNet.

    Hierarchy:
        trunk0:
            xif0_0
            xif0_1
            ...
        trunk1:
            xif1_0
            xif1_1
            ...
        ...

    Output features:
        The outputs from each "stage", i.e. trunkX.
    """

    def __init__(self, cfg):
        super().__init__()

        self.vgg = make_layers(cfgs['vgg16'],batch_norm=True)

        self._initialize_weights()
        # self.stage_names_index = {'vgg1':3, 'vgg2':8 , 'vgg3':15, 'vgg4':22, 'vgg5':29}
        _out_feature_channels = [64, 128, 256, 512, 512]
        _out_feature_strides = [2, 4, 8, 16, 32]

        # For images
        self.stages = [nn.Sequential(*list(self.vgg._modules.values())[0:7]),\
                    nn.Sequential(*list(self.vgg._modules.values())[7:14]),\
                    nn.Sequential(*list(self.vgg._modules.values())[14:24]),\
                    nn.Sequential(*list(self.vgg._modules.values())[24:34]),\
                    nn.Sequential(*list(self.vgg._modules.values())[34:]),]
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._stage_names = []

        for i, stage in enumerate(self.stages):
            name = "vgg{}".format(i)
            self.add_module(name, stage)
            self._stage_names.append(name)
            self._out_feature_channels[name] = _out_feature_channels[i]
            self._out_feature_strides[name] = _out_feature_strides[i]

        self._out_features = self._stage_names

        del self.vgg

        # assume stage channel dims: [64, 128, 256, 512, 512]
        # self.depth = WindowedDepthXAttn(embed_dim=512)

    def forward(self, x_rgb, depth):
        feats = {}
        x = x_rgb
        
        d = depth[:, 0:1, :, :].float()
        d = (d - d.amin((2,3), True)) / (d.amax((2,3), True) - d.amin((2,3), True) + 1e-8)
        x = x * (1 + d)

        for name, stage in zip(self._stage_names, self.stages):
            x = stage(x)
            feats[name] = x
        return feats

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_backbone(cfg, _):
    return vgg_backbone(cfg)


@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_fpn_backbone(cfg, _):

    bottom_up = vgg_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
    )

    return backbone
