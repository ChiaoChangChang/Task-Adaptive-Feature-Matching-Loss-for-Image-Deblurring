"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from MPRNet import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx

from torchvision import transforms
from ignite.metrics import PSNR
from ssim import SSIM
import lpips

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, dataset, 'test')
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir = os.path.join(args.result_dir, dataset)
utils.mkdir(result_dir)

score_dir = os.path.join(args.result_dir, dataset + '_score')
utils.mkdir(score_dir)

to_pil_image = transforms.ToPILImage()
psnr = PSNR(data_range=1.0)
per_img_psnr = PSNR(data_range=1.0)
loss_fn_alex = lpips.LPIPS(net='alex')

counter = 0
total_ssim = 0
total_lpips = 0

img_psnr = []
img_lpips = []

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        target = data_test[0]
        input_ = data_test[1].cuda()
        filenames = data_test[2]

        # Padding in case images are not multiples of 8
        if dataset == 'RealBlur-J' or dataset == 'RealBlur-R':
            factor = 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)
        restored = torch.clamp(restored[0],0,1)

        # Unpad images to original dimensions
        if dataset == 'RealBlur-J' or dataset == 'RealBlur-R':
            restored = restored[:,:,:h,:w]

        # process image
        restored = restored.cpu().detach()
        
        restored_pil = to_pil_image(restored.squeeze(0))
        target_pil = to_pil_image(target.squeeze(0))

        restored_norm = restored * 2.0 - 1.0
        target_norm = target * 2.0 - 1.0

        # metrics
        counter += 1
        psnr.update((restored, target))
        total_ssim += SSIM(restored_pil).cw_ssim_value(target_pil)
        lpips_val = loss_fn_alex(restored_norm, target_norm)
        total_lpips += lpips_val

        # per image information
        per_img_psnr.reset()
        per_img_psnr.update((restored, target))
        psnr_val = f'{per_img_psnr.compute() : .6f}'
        lpips_val = f'{lpips_val.item() : .6f}'
        
        img_psnr.append([filenames[0] + '.png', psnr_val])
        img_lpips.append([filenames[0] + '.png', lpips_val])
        
#         restored = restored.permute(0, 2, 3, 1).numpy()
        
#         for batch in range(len(restored)):
#             restored_img = img_as_ubyte(restored[batch])
#             utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
        
result_str = f'PSNR: {psnr.compute() : .6f}, SSIM: {total_ssim / counter : .6f}, LPIPS: {total_lpips.item() / counter : .6f}'
print(result_str)

with open(os.path.join(score_dir, 'result.txt'), 'w') as f:
    f.write(result_str + '\n')

with open(os.path.join(score_dir, 'psnr_val.csv'), 'w') as f:
    f.write('filename,PSNR\n')
    for r in img_psnr:
        f.write(f'{r[0]},{r[1]}\n')

with open(os.path.join(score_dir, 'lpips_val.csv'), 'w') as f:
    f.write('filename,LPIPS\n')
    for r in img_lpips:
        f.write(f'{r[0]},{r[1]}\n')
        
# python test.py --dataset GoPro --weights clip_mlp_layer2/Deblurring/models/MPRNet/model_latest.pth --result_dir clip_mlp_layer2_gopro --gpus 0
