
from SSIM_loss import *

from VGG_loss import *
from torchvision import models

from scipy.ndimage import gaussian_filter
import numpy as np
import math


class combinedloss(nn.Module):
    def __init__(self, config):
        super(combinedloss, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG model is loaded")
        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config['device'])
        self.l1loss = nn.L1Loss().to(config['device'])
        # self.l1loss = nn.L().to(config['device'])

    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.mseloss(out, label)

        # ssim_loss = compute_ssim(out.cpu(), label.cpu())

        # ssim_loss = tf.constant(1, dtype=tf.float32) - tf.reduce_mean(tf.image.ssim(a, b, max_val=1, filter_size=13))
        # 1 - torch.mean(torch.image.ssim(out, label, max_val=1, filter_size=13))

        ssim_loss = 1-torch.mean(ssim(out, label)) ###
        # ssim_loss = 1 - torch.mean(ms_ssim(out, label))  ### 这个是多尺度SSIM，效果不如SSIM

        vgg_loss = self.l1loss(inp_vgg, label_vgg)
        # total_loss = mse_loss + vgg_loss
        total_loss = mse_loss + vgg_loss + ssim_loss
        return total_loss, mse_loss, vgg_loss, ## ssim_loss
