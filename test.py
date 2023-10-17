import numpy as np
import torchvision
from torchvision import transforms
import argparse
import time
from tqdm import tqdm
from model import *
from dataloader import myDataSet
from metrics_calculation import *
import os

import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
import imageio
import PIL.Image
import scipy.misc
from utils.uqim_utils import getUIQM
from utils.ssm_psnr_utils import getSSIM, getPSNR
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
__all__ = [
    "test",
    "setup",
    "testing",
]

@torch.no_grad()
def test(config, test_dataloader, test_model):
    test_model.eval()
    for i, (img, _, name) in enumerate(test_dataloader):
        with torch.no_grad():
            img = img.to(config.device)
            generate_img = test_model(img)
            torchvision.utils.save_image(generate_img, config.output_images_path + name[0])

def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    model = torch.load(config.snapshot_path).to(config.device)
    transform = transforms.Compose([transforms.Resize((config.resize,config.resize)),transforms.ToTensor()]) ## 这里修改图像尺寸
    # transform = transforms.Compose([transforms.Resize((1295, 1800)), transforms.ToTensor()])
    test_dataset = myDataSet(config.test_images_path,None,transform, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size = config.batch_size,shuffle = False)
    print("Test Dataset Reading Completed.")
    return test_dataloader, model

def testing(config):
    ds_test, model = setup(config)
    test(config, ds_test, model)


def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f == gen_f):
            # assumes same filenames
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            # get psnt on L channel (SOTA norm)
            r_im = r_im.convert("L");
            g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)


def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)

if __name__ == '__main__':
    models = "./snapshots/model_epoch_" + str(99) + ".ckpt"
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot_path', type=str, default=models, help='one')
    parser.add_argument('--test_images_path', type=str, default="./data/raw/", help='path of input images(underwater images) for testing default:./data/input/')
    parser.add_argument('--output_images_path', type=str, default='./data/output/', help='path to save generated image.')
    parser.add_argument('--batch_size', type=int, default=1, help="default : 1")
    parser.add_argument('--resize', type=int, default=256, help="resize images, default:resize images to 256*256")
    parser.add_argument('--calculate_metrics', type=bool, default=False, help="calculate PSNR, SSIM and UIQM on test images")
    parser.add_argument('--label_images_path', type=str, default="./data/label/", help='path of label images(clear images) default:./data/label/')

    print("-------------------testing---------------------")
    config = parser.parse_args()
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)

    start_time = time.time()
    testing(config)
    print("total testing time", time.time() - start_time) ## more images for PFS test
    # ### 以下是做测试
    GEN_im_dir = "./data/output/"
    GTr_im_dir = "./data/GT/"
    gen_uqims = measure_UIQMs(GEN_im_dir)
    print("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))
    ### compute SSIM and PSNR
    SSIM_measures, PSNR_measures = SSIMs_PSNRs(GTr_im_dir, GEN_im_dir)
    print("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
    print("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))


