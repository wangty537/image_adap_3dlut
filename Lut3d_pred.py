import argparse
import math

import pytorch_ssim
import torch
import os
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
from PIL import Image
from torch.autograd import Variable

from Lut3d_model import LUT0, LUT1, LUT2, classifier, TV3, generator_train, get_loss, criterion_pixelwise, generator_eval, weights_init_normal_classifier
from Lut3d import data_loader_train, data_loader_test
from Lut3d_para import opt
import colour




def trilinear_forward_pixel(lut, image, dim=33):
    height, width, channels = image.shape
    output_size = height * width
    binsize = 1.0000001 / (dim - 1)
    image = np.reshape(image, [-1, 3])
    lut = np.reshape(lut, [-1, 3])
    for index in range(output_size):
        r = image[index, 0]
        g = image[index, 1]
        b = image[index, 2]
        r_id = np.floor(r / binsize).astype(np.int32)
        g_id = np.floor(g / binsize).astype(np.int32)
        b_id = np.floor(b / binsize).astype(np.int32)

        r_d = np.fmod(r, binsize) / binsize
        g_d = np.fmod(g, binsize) / binsize
        b_d = np.fmod(b, binsize) / binsize

        id000 = r_id + g_id * dim + b_id * dim * dim
        id100 = r_id + 1 + g_id * dim + b_id * dim * dim
        id010 = r_id + (g_id + 1) * dim + b_id * dim * dim
        id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim
        id001 = r_id + g_id * dim + (b_id + 1) * dim * dim
        id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim
        id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim
        id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim

        w000 = (1 - r_d) * (1 - g_d) * (1 - b_d)
        w100 = r_d * (1 - g_d) * (1 - b_d)
        w010 = (1 - r_d) * g_d * (1 - b_d)
        w110 = r_d * g_d * (1 - b_d)
        w001 = (1 - r_d) * (1 - g_d) * b_d
        w101 = r_d * (1 - g_d) * b_d
        w011 = (1 - r_d) * g_d * b_d
        w111 = r_d * g_d * b_d

        image[index, 0] = w000 * lut[id000, 0] + w100 * lut[id100, 0] +\
                          w010 * lut[id010, 0] + w110 * lut[id110, 0] +\
                          w001 * lut[id001, 0] + w101 * lut[id101, 0] +\
                          w011 * lut[id011, 0] + w111 * lut[id111, 0]

        image[index, 1] = w000 * lut[id000, 1] + w100 * lut[id100, 1] +\
                          w010 * lut[id010, 1] + w110 * lut[id110, 1] +\
                          w001 * lut[id001, 1] + w101 * lut[id101, 1] +\
                          w011 * lut[id011, 1] + w111 * lut[id111, 1]

        image[index, 2] = w000 * lut[id000, 2] + w100 * lut[id100, 2] +\
                          w010 * lut[id010, 2] + w110 * lut[id110, 2] +\
                          w001 * lut[id001, 2] + w101 * lut[id101, 2] +\
                          w011 * lut[id011, 2] + w111 * lut[id111, 2]
        return image.reshape([height, width, channels])
def trilinear_forward(img, lut, lut_size=33):
    """
    :param img: h * w * channel numpy array, float
    :param lut: 3d lut, size[-1, 3]
    :param lut_size: default:33
    :return: img_lut
    """
    bin_size = 1.0000001 / (lut_size - 1)
    dim = lut_size
    # for i in range(h):
    #     for j in range(w):
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    r_id = np.floor(r / bin_size).astype(np.int32)
    g_id = np.floor(g / bin_size).astype(np.int32)
    b_id = np.floor(b / bin_size).astype(np.int32)
    r_d = np.fmod(r, bin_size) / bin_size
    g_d = np.fmod(g, bin_size) / bin_size
    b_d = np.fmod(b, bin_size) / bin_size

    id000 = r_id + g_id * dim + b_id * dim * dim
    id100 = r_id + 1 + g_id * dim + b_id * dim * dim
    id010 = r_id + (g_id + 1) * dim + b_id * dim * dim
    id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim
    id001 = r_id + g_id * dim + (b_id + 1) * dim * dim
    id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim
    id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim
    id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim

    w000 = (1 - r_d) * (1 - g_d) * (1 - b_d)
    w100 = r_d * (1 - g_d) * (1 - b_d)
    w010 = (1 - r_d) * g_d * (1 - b_d)
    w110 = r_d * g_d * (1 - b_d)
    w001 = (1 - r_d) * (1 - g_d) * b_d
    w101 = r_d * (1 - g_d) * b_d
    w011 = (1 - r_d) * g_d * b_d
    w111 = r_d * g_d * b_d

    w000 = w000[..., None]
    w100 = w100[..., None]
    w010 = w010[..., None]
    w110 = w110[..., None]
    w001 = w001[..., None]
    w101 = w101[..., None]
    w011 = w011[..., None]
    w111 = w111[..., None]
    rgb = w000 * lut[id000] + w100 * lut[id100] + \
        w010 * lut[id010] + w110 * lut[id110] + \
        w001 * lut[id001] + w101 * lut[id101] + \
        w011 * lut[id011] + w111 * lut[id111]

    return rgb
# def generate_LUT(img):
#     pred = classifier(img).squeeze()
#
#     LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
#
#     return LUT


# ----------
#  test
# ----------
# read image and transform to tensor
# image_path = r'demo_images/sRGB/a1629.jpg'
# img = Image.open(image_path)
# print(img.size)
# img = TF.to_tensor(img).type(torch.Tensor)
# img_my = img.copy()
# img = img.unsqueeze(0)
#
# LUT = generate_LUT(img)
def convert_im(ret, filename):
    ndarr = ret.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
    return ndarr
def show_im(real_A, real_B, fake_B):
    file1 = 'a.png'
    file2 = 'b.png'
    file3 = 'c.png'
    print(real_A.dtype, real_B.dtype, fake_B.dtype)
    a = convert_im(real_A, file1)
    b = convert_im(real_B, file2)
    c = convert_im(fake_B, file3)
    d = np.hstack((a, b, c))
    d = d[:, :, ::-1]
    cv2.imshow('dd', d)
    cv2.waitKey(0)
    return a, b, c
def pred_model(data_loader_t):
    # Load pretrained models
    epoch = 70
    LUTs = torch.load("LUTs4_%d.pth" % epoch)
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    # LUT3.load_state_dict(LUTs["3"])
    # LUT4.load_state_dict(LUTs["4"])
    LUT0.eval()
    LUT1.eval()
    LUT2.eval()
    # LUT3.eval()
    # LUT4.eval()
    classifier.load_state_dict(torch.load("classifier4_%d.pth" % epoch))
    classifier.eval()

    psnr_li = []
    ssim_li = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader_t):
            real_A, real_B, img_names = batch
            fake_B, weights_norm, LUT = generator_eval(real_A)
            print(LUT.numpy().shape)
            mse = criterion_pixelwise(fake_B, real_B)
            psnr = 10 * math.log10(1 / mse.item())
            ssim_value = pytorch_ssim.ssim(fake_B, real_B).item()

            psnr_li.append(psnr)
            ssim_li.append(ssim_value)
            a, b, c = show_im(real_A, real_B, fake_B)
            deltaE = colour.delta_E(b, c).mean()
            print(" [PSNR: %f] [SSIM: %f] [deltaE: %f] \n" % (psnr, ssim_value, deltaE))
    return psnr_li, ssim_li
def show_im1():
    n = 33
    a1 = 1.85
    a2 = -0.09
    a3 = -0.91
    LUT1 = np.loadtxt(r'G:\github\tmp\Image-Adaptive-3DLUT-master\Image-Adaptive-3DLUT-master\visualization_lut\learned_LUT_234_1.txt')
    LUT2 = np.loadtxt(r'G:\github\tmp\Image-Adaptive-3DLUT-master\Image-Adaptive-3DLUT-master\visualization_lut\learned_LUT_234_2.txt')
    LUT3 = np.loadtxt(r'G:\github\tmp\Image-Adaptive-3DLUT-master\Image-Adaptive-3DLUT-master\visualization_lut\learned_LUT_234_3.txt')
    LUT = LUT1 * a1 + LUT2 * a2 + LUT3 * a3;
    r = LUT[:n**3]
    r = np.reshape(r,[n,n,n])
    g = LUT[n**3:n**3*2]
    g = np.reshape(g,[n,n,n])
    b = LUT[n**3*2:n**3*3]
    b = np.reshape(b,[n,n,n])

    LUT_my = np.reshape(LUT, [3, -1]).T
    print('LUT_my', LUT_my)
    result2 = trilinear_forward(np.array(img)/255, LUT_my, lut_size=33)
    result3 = trilinear_forward_pixel(LUT_my, np.array(img)/255, dim=33)
    # save image
    # ndarr = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # im = Image.fromarray(ndarr)
    # im.save('%s/result.jpg' % opt.output_dir, quality=95)


    a = np.array(img)[:,:,::-1]
    b = np.clip(result2, 0, 1)*255
    b = b.astype(np.uint8)

    c = np.clip(result3, 0, 1)*255
    c = b.astype(np.uint8)

    cv2.imwrite('b.png', b)
    cv2.imwrite('c.png', c)
    r = np.hstack((a, b, c))
    cv2.namedWindow("rrr", 0)
    cv2.imshow('rrr', r)
    cv2.waitKey(0)

if __name__ == "__main__":
    pred_model(data_loader_test)