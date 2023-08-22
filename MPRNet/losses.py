import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.cuda.amp import autocast
from CLIP_MLP.model import CLIPMLP

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class ClipLoss(nn.Module):
    def __init__(self, backbone='ViT-B/16'):
        super(ClipLoss, self).__init__()
        self.clip_model, _ = clip.load(backbone, device='cuda', jit=False)
        self.clip_model.eval()
        self.visual = self.clip_model.visual
        self.image_size = self.visual.input_resolution

    def extract_feature(self, image):
        v = self.visual
        x = image.type(v.conv1.weight.dtype)
        x = F.interpolate(x, size=self.image_size, mode='bicubic')
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

        feat = xfinal.type(torch.float32)

        return feat

    def forward(self, x, y):
        fx = self.extract_feature(x)
        fy = self.extract_feature(y)
        fy_no_grad = fy.detach()
        loss = F.mse_loss(fx, fy_no_grad)
        return loss

class MLPLoss(nn.Module):
    def __init__(self, backbone='ViT-B/16'):
        super(MLPLoss, self).__init__()
        self.model = CLIPMLP(clip_model='ViT-B/16', clip_layer='layer_2',
                             layer_width=1024, use_batchnorm=True, is_inference=True, weight_path='CLIP_MLP/checkpoints/checkpoints_16_block6_lw1024_lr1e-4/model_best.pth')
        self.model = self.model.to('cuda')
        
    def forward(self, x, y):
        fxs = self.model(x)
        fys = self.model(y)
        losses = []
        for i in range(len(fxs)):
            fx = fxs[i]
            fy_no_grad = fys[i].detach()
            loss = F.mse_loss(fx, fy_no_grad)
            losses.append(loss)
        loss = sum(losses) / len(fxs)
        return loss
