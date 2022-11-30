# TRAINING
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from timm.models.vision_transformer import _cfg
from functools import partial
from PVT import PyramidVisionTransformer

def softmax(data):
    for i in range(data.shape[0]):
        f = data[i,:].reshape (data.shape[1])
        data[i,:] = torch.exp(f) / torch.sum(torch.exp(f))
    return data

class SpatialSoftmax3D(torch.nn.Module):
    def __init__(self, height, width, depth, channel, lim=[0., 1., 0., 1., 0., 1.], temperature=None, data_format='NCHWD'):
        super(SpatialSoftmax3D, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.depth = depth
        self.channel = channel
        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.
        pos_y, pos_x, pos_z = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height),
            np.linspace(lim[4], lim[5], self.depth))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width * self.depth)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width * self.depth)).float()
        pos_z = torch.from_numpy(pos_z.reshape(self.height * self.width * self.depth)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)
    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWDC':
            feature = feature.transpose(1, 4).tranpose(2, 4).tranpose(3,4).reshape(-1, self.height * self.width * self.depth)
        else:
            feature = feature.reshape(-1, self.height * self.width * self.depth)
        softmax_attention = feature
        # softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        heatmap = softmax_attention.reshape(-1, self.channel, self.height, self.width, self.depth)

        eps = 1e-6
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xyz.reshape(-1, self.channel, 3)
        return feature_keypoints, heatmap

def pvt_medium6DOF(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        img_size = 96,
        patch_size=2, in_chans = 20, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model

class UNet6DOF_medium(nn.Module):
  def __init__(self):
    self.x = 2
    self.k = 4
    self.m = 4
    self.n = 4
    super().__init__()
    self.pvtMedium = pvt_medium6DOF()
    self.conv_00 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(5,5,5)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32))
    self.conv_01 = nn.Sequential(
            nn.Conv3d(32, 21, kernel_size=(5,5,5)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(21),
            nn.MaxPool3d(kernel_size=2),
            nn.Sigmoid())
    self.convTrans_00 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(2,2,2),stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm3d(256))
    self.convTrans_01 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(2,2,2),stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128))
    self.convTrans_02 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(2,2,2),stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64))
    
    
  def forward(self, input):
    out, features = self.pvtMedium.forward_features(input)
    out = out.reshape(20, 6, 6, 512)
    out = out.transpose(1, 3).transpose(2, 3)
    out = out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3], 1)
    out = out.repeat(1,1,1,1, self.x)
    out = self.convTrans_00(out)
    
    fea1 = features[2].reshape(20, 12, 12, 256).transpose(1, 3).transpose(2, 3)
    fea1 = fea1.reshape(fea1.shape[0], fea1.shape[1], fea1.shape[2], fea1.shape[3], 1).repeat(1,1,1,1,self.k)
    out = torch.cat((out, fea1), axis=-1)
    out = self.convTrans_01(out)

    fea2 = features[1].reshape(-1, 24, 24, 128).transpose(1, 3).transpose(2, 3)
    fea2 = fea2.reshape(fea2.shape[0], fea2.shape[1], fea2.shape[2], fea2.shape[3], 1).repeat(1,1,1,1,self.m)
    out = torch.cat((out, fea2), axis=-1)
    out = self.convTrans_02(out)

    fea3 = features[0].reshape(-1, 48, 48, 64).transpose(1, 3).transpose(2, 3)
    fea3 = fea3.reshape(fea3.shape[0], fea3.shape[1], fea3.shape[2], fea3.shape[3], 1).repeat(1,1,1,1,self.n)
    out = torch.cat((out, fea3), axis=-1)

    print()
    out = self.conv_00(out)
    out = self.conv_01(out)

    return out

# input = torch.rand((20,20,96,96))
# model = UNet6DOF_medium()
# output = model(input)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print (pytorch_total_params)
# print(output.shape)
