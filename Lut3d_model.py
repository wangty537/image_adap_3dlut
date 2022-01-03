
import argparse
import itertools
import math
import time
import datetime
import sys

import torch.nn as nn
import torch.nn.functional as TF
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Lut3d_trilinear import trilinear_forward_tensor



class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt", 'r')
        LUT = file.readlines()
        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k
                    x = LUT[n].split()
                    self.LUT[0, i, j, k] = float(x[0])
                    self.LUT[1, i, j, k] = float(x[1])
                    self.LUT[2, i, j, k] = float(x[2])
        self.LUT = nn.Parameter(torch.tensor(self.LUT))

    def forward(self, x):

        return trilinear_forward_tensor(x, self.LUT, lut_size=33)


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        self.LUT = nn.Parameter(torch.tensor(self.LUT))

    def forward(self, x):
        return trilinear_forward_tensor(x, self.LUT, lut_size=33)

def discriminator_block(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))
    return layers


class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)

class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn



LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
TV3 = TV_3D()
classifier = Classifier()
# Loss functions
criterion_pixelwise = torch.nn.MSELoss()
def generator_train(img):

    pred = classifier(img).squeeze()
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    gen_A0 = LUT0(img)
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)
    # gen_A3 = LUT3(img)
    # gen_A4 = LUT4(img)

    weights_norm = torch.mean(pred ** 2)
    combine_A = img.new(img.size())
    for b in range(img.size(0)):
        combine_A[b, :, :, :] = pred[b, 0] * gen_A0[b, :, :, :] + \
                                pred[b, 1] * gen_A1[b, :, :, :] + \
                                pred[b, 2] * gen_A2[b, :, :, :]

    return combine_A, weights_norm
# 应该等价于generator_train
def generator_eval(img):

    pred = classifier(img).squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    weights_norm = torch.mean(pred ** 2)
    combine_A = trilinear_forward_tensor(img, LUT, lut_size=33)
    return combine_A, weights_norm, LUT

def get_loss(fake_B, real_B, LUT0, LUT1, LUT2, weights_norm):


    # Pixel-wise loss
    mse = criterion_pixelwise(fake_B, real_B)

    tv0, mn0 = TV3(LUT0)
    tv1, mn1 = TV3(LUT1)
    tv2, mn2 = TV3(LUT2)
    # tv3, mn3 = TV3(LUT3)
    # tv4, mn4 = TV3(LUT4)
    tv_cons = tv0 + tv1 + tv2  # + tv3 + tv4
    mn_cons = mn0 + mn1 + mn2  # + mn3 + mn4

    loss = mse + 0.0001 * (weights_norm + tv_cons) + 10.0 * mn_cons
    return mse, loss

def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)