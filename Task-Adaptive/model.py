import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch.cuda.amp import autocast

class CLIPMLP(nn.Module):
    def __init__(self, clip_model='ViT-B/16', clip_layer='layer_final', layer_width=1024, use_batchnorm=True, is_inference=False, weight_path=''):
        super().__init__()

        self.layer_to_input_size = {
            'layer_2': 151296,
            'layer_final': 512
        }

        if clip_layer not in self.layer_to_input_size.keys():
            raise Exception(f'CLIP layer "{clip_layer}" undefined')

        self.clip_layer = clip_layer
        self.input_size = self.layer_to_input_size[clip_layer]

        clip_model, self.preprocess = clip.load(clip_model, device='cuda', jit=False)
        self.clip = clip_model.visual
        for param in self.clip.parameters():
            param.requires_grad = False

        if use_batchnorm:
            self.mlp = nn.Sequential(
                nn.Linear(self.input_size, layer_width),
                nn.BatchNorm1d(layer_width),
                nn.ReLU(),
                nn.Linear(layer_width, layer_width),
                nn.BatchNorm1d(layer_width),
                nn.ReLU(),
                nn.Linear(layer_width, layer_width)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.input_size, layer_width),
                nn.ReLU(),
                nn.Linear(layer_width, layer_width),
                nn.ReLU(),
                nn.Linear(layer_width, layer_width)
            )
        
        if is_inference:
            if not os.path.exists(weight_path):
                raise Exception(f'Model weight path {weight_path} does not exist')
            else:
                self.load_state_dict(torch.load(weight_path))
                self.clip.eval()
                self.mlp.eval()
                self.eval()
                print(f'Successfully loaded model from {weight_path}')

    def clip_layer_2_fwd(self, x):
        x = x.type(self.clip.conv1.weight.dtype)
        x = F.interpolate(x, size=self.clip.input_resolution, mode='bicubic')
        x = self.clip.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.positional_embedding.to(x.dtype)
        x = self.clip.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i, b in enumerate(self.clip.transformer.resblocks):
            x = b(x)
            if i == 3:
                xs = torch.flatten(x.permute(1, 0, 2), start_dim=1).type(torch.float32)

        return xs

    def clip_layer_final_fwd(self, x):
        x = x.type(self.clip.conv1.weight.dtype)
        x = F.interpolate(x, size=self.clip.input_resolution, mode='bicubic')
        x = self.clip(x)
        x = x.type(torch.float32)
        return x

    def forward(self, x):
        with autocast():
            if self.clip_layer == 'layer_2':
                x = self.clip_layer_2_fwd(x)
            else:
                x = self.clip_layer_final_fwd(x)

            x = self.mlp(x)

        return x

if __name__ == '__main__':
    model = CLIPMLP()
    print(model)
