
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
# from basicsr.archs.vgg_arch import VGGFeatureExtractor
# from basicsr.utils.registry import LOSS_REGISTRY
import clip
from CLIP_MLP.model import CLIPMLP

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class ClipLoss(nn.Module):
    def __init__(self,
                 loss_weight1=0.5, reduction='mean', toY=False,  # PSNR Loss
                 loss_weight2=100.0, backbone='ViT-B/16'):       # CLIP Loss
        super(ClipLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight1 = loss_weight1
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

        self.loss_weight2 = loss_weight2
        self.clip_model, _ = clip.load(backbone, device="cuda", jit=False)
        self.clip_model.eval()
        self.visual = self.clip_model.visual

    def extract_feature(self, image):		
        v = self.visual
        x = image.type(v.conv1.weight.dtype)
        x = F.interpolate(x, size=v.input_resolution, mode='bicubic')
        #xfinal = v(x)
        x = v.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i, b in enumerate(v.transformer.resblocks):
            x = b(x)
            if i == 3:
                feat = torch.flatten(x.permute(1, 0, 2), start_dim=1).type(torch.float32)


        #feat = xfinal.type(torch.float32)

        return feat

    def psnr_loss(self, pred, target):      
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        
        return self.loss_weight1 * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    
    def clip_loss(self, pred, target):
        f_fake = self.extract_feature(pred)
        f_real = self.extract_feature(target)
        f_real_no_grad = f_real.detach()
        loss = F.mse_loss(f_fake, f_real_no_grad)

        return self.loss_weight2 * loss

    def forward(self, pred, target):
        loss1 = self.psnr_loss(pred, target)
        loss2 = self.clip_loss(pred, target)
        # print("Loss1: %.6f \n Loss2: %.6f"%(loss1, loss2))\

        return  loss1 + loss2


class MixLoss(nn.Module):
    def __init__(self,
                 loss_weight1=0.5, reduction='mean', toY=False,  # PSNR Loss
                 loss_weight2=100.0, backbone='ViT-B/16', layer='layer_2'):       # CLIP Loss

        super(MixLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight1 = loss_weight1
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

        self.loss_weight2 = loss_weight2
        self.model = CLIPMLP(clip_model=backbone, clip_layer=layer)
        self.model.load_state_dict(torch.load('CLIP_MLP/checkpoints/ViT-B-16/16_layer_2.pth'))
        self.model.clip.eval()
        self.model.mlp.eval()
        self.model.eval()
        self.model = self.model.to('cuda')

    def psnr_loss(self, pred, target):		
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        
        return self.loss_weight1 * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    
    def clip_loss(self, pred, target):
        fx = self.model(pred)
        fy = self.model(target)
        fy_no_grad = fy.detach()
        loss = F.mse_loss(fx, fy_no_grad)

        return self.loss_weight2 * loss

    def forward(self, pred, target):
        loss1 = self.psnr_loss(pred, target)
        loss2 = self.clip_loss(pred, target)
        # print("Loss1: %.6f \n Loss2: %.6f"%(loss1, loss2))\

        return  loss1 + loss2

# @LOSS_REGISTRY.register()
# class PerceptualLoss(nn.Module):
#     """Perceptual loss with commonly used style loss.
#     Args:
#         layer_weights (dict): The weight for each layer of vgg feature.
#             Here is an example: {'conv5_4': 1.}, which means the conv5_4
#             feature layer (before relu5_4) will be extracted with weight
#             1.0 in calculating losses.
#         vgg_type (str): The type of vgg network used as feature extractor.
#             Default: 'vgg19'.
#         use_input_norm (bool):  If True, normalize the input image in vgg.
#             Default: True.
#         range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
#             Default: False.
#         perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
#             loss will be calculated and the loss will multiplied by the
#             weight. Default: 1.0.
#         style_weight (float): If `style_weight > 0`, the style loss will be
#             calculated and the loss will multiplied by the weight.
#             Default: 0.
#         criterion (str): Criterion used for perceptual loss. Default: 'l1'.
#     """

#     def __init__(self,
#                  layer_weights,
#                  vgg_type='vgg19',
#                  use_input_norm=True,
#                  range_norm=False,
#                  perceptual_weight=1.0,
#                  style_weight=0.,
#                  criterion='l1'):
#         super(PerceptualLoss, self).__init__()
#         self.perceptual_weight = perceptual_weight
#         self.style_weight = style_weight
#         self.layer_weights = layer_weights
#         self.vgg = VGGFeatureExtractor(
#             layer_name_list=list(layer_weights.keys()),
#             vgg_type=vgg_type,
#             use_input_norm=use_input_norm,
#             range_norm=range_norm)

#         self.criterion_type = criterion
#         if self.criterion_type == 'l1':
#             self.criterion = torch.nn.L1Loss()
#         elif self.criterion_type == 'l2':
#             self.criterion = torch.nn.L2loss()
#         elif self.criterion_type == 'fro':
#             self.criterion = None
#         else:
#             raise NotImplementedError(f'{criterion} criterion has not been supported.')

#     def forward(self, x, gt):
#         """Forward function.
#         Args:
#             x (Tensor): Input tensor with shape (n, c, h, w).
#             gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
#         Returns:
#             Tensor: Forward results.
#         """
#         # extract vgg features
#         x_features = self.vgg(x)
#         gt_features = self.vgg(gt.detach())

#         # calculate perceptual loss
#         if self.perceptual_weight > 0:
#             percep_loss = 0
#             for k in x_features.keys():
#                 if self.criterion_type == 'fro':
#                     percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
#                 else:
#                     percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
#             percep_loss *= self.perceptual_weight
#         else:
#             percep_loss = None

#         # calculate style loss
#         if self.style_weight > 0:
#             style_loss = 0
#             for k in x_features.keys():
#                 if self.criterion_type == 'fro':
#                     style_loss += torch.norm(
#                         self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
#                 else:
#                     style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
#                         gt_features[k])) * self.layer_weights[k]
#             style_loss *= self.style_weight
#         else:
#             style_loss = None

#         return percep_loss, style_loss

#     def _gram_mat(self, x):
#         """Calculate Gram matrix.
#         Args:
#             x (torch.Tensor): Tensor with shape of (n, c, h, w).
#         Returns:
#             torch.Tensor: Gram matrix.
#         """
#         n, c, h, w = x.size()
#         features = x.view(n, c, w * h)
#         features_t = features.transpose(1, 2)
#         gram = features.bmm(features_t) / (c * h * w)
#         return gram