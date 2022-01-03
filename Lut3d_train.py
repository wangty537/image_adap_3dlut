
import itertools
import math
import sys

import pytorch_ssim
import torch
from PIL import Image
from torch.autograd import Variable

from Lut3d_model import LUT0, LUT1, LUT2, classifier, TV3, generator_train, get_loss, criterion_pixelwise, generator_eval, weights_init_normal_classifier
from Lut3d import data_loader_train, data_loader_test
from Lut3d_para import opt

# devices setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
print(device)
LUT0.to(device)
LUT1.to(device)
LUT2.to(device)
classifier.to(device)
criterion_pixelwise.to(device)
TV3.to(device)
TV3.weight_r = TV3.weight_r.type(Tensor)
TV3.weight_g = TV3.weight_g.type(Tensor)
TV3.weight_b = TV3.weight_b.type(Tensor)
# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(classifier.parameters(), LUT0.parameters(), LUT1.parameters(), LUT2.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)) #, LUT3.parameters(), LUT4.parameters()

def train(epoch, n_epochs, data_loader_t):
    psnr_avg = 0
    ssim_value = 0
    for i, batch in enumerate(data_loader_t):
        # Model inputs
        real_A, real_B, img_names = batch
        real_A = Variable(batch[0].type(Tensor))
        real_B = Variable(batch[1].type(Tensor))

        # forward
        optimizer_G.zero_grad()
        fake_B, weights_norm = generator_train(real_A)
        mse, loss = get_loss(fake_B, real_B, LUT0, LUT1, LUT2, weights_norm)
        # backward
        loss.backward()
        # update
        optimizer_G.step()

        # log
        psnr_avg += 10 * math.log10(1 / mse.item())
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [psnr: %f]"
            % (
                epoch, n_epochs, i, len(data_loader_t), psnr_avg / (i + 1)
            )
        )


# import numpy as np
#
# import matplotlib.pyplot as plt
# def show_im(img_t):
#     plt.figure()
#     img = img_t.numpy()
#     img = np.transpose(img, (1, 2, 0))
#     plt.imshow(img)
def data_test(epoch, data_loader_t):
    max_psnr = 0
    psnr_avg = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader_t):
            # real_A, real_B, img_names = batch
            real_A = Variable(batch[0].type(Tensor))
            real_B = Variable(batch[1].type(Tensor))
            fake_B, weights_norm = generator_eval(real_A)
            mse = criterion_pixelwise(fake_B, real_B)
            psnr = 10 * math.log10(1 / mse.item())
            psnr_avg += psnr

            # if i > 20:
            #     break
        psnr_avg /= (i + 1)
        if psnr_avg > max_psnr:
            max_psnr = psnr_avg
            max_epoch = epoch
            sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (psnr_avg, max_psnr, max_epoch))
    return psnr_avg



import cv2
import numpy as np

if __name__ == "__main__":
    print(len(data_loader_train), len(data_loader_test))
    # data_test(0, data_loader_train)
    # data_test(0, data_loader_test)
    if opt.epoch >= 0:
        # Load pretrained models
        print("LUTs4_%d.pth" % opt.epoch)
        LUTs = torch.load("LUTs4_%d.pth" % opt.epoch)
        LUT0.load_state_dict(LUTs["0"])
        LUT1.load_state_dict(LUTs["1"])
        LUT2.load_state_dict(LUTs["2"])
        # LUT3.load_state_dict(LUTs["3"])
        # LUT4.load_state_dict(LUTs["4"])
        classifier.load_state_dict(torch.load("classifier4_%d.pth" %  opt.epoch))
    else:
        # Initialize weights
        classifier.apply(weights_init_normal_classifier)
        torch.nn.init.constant_(classifier.model[16].bias.data, 1.0)

    for epoch in range(opt.epoch, opt.n_epochs):
        train(epoch, opt.n_epochs, data_loader_train)
        data_test(epoch, data_loader_test)

        if epoch % 10 == 0:
            # Save model checkpoints
            LUTs = {"0": LUT0.state_dict(), "1": LUT1.state_dict(),
                    "2": LUT2.state_dict()}  # ,"3": LUT3.state_dict(),"4": LUT4.state_dict()
            torch.save(LUTs, "LUTs4_%d.pth" % epoch)
            torch.save(classifier.state_dict(), "classifier4_%d.pth" % (epoch))