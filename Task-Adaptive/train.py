import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SharpBlurImageDataset, get_train_val_paths
from model import CLIPMLP
from utils import AverageMeter

# config
DATASET_ROOT = 'dataset'
CHECKPOINT_ROOT = 'checkpoints'
HISTORY_ROOT = 'history'
VAL_SIZE = 0.1
RANDOM_SEED = 1
IMG_SIZE = 224
CLIP_MODEL = 'ViT-B/16'
CLIP_LAYER = 'layer_2'
USE_BATCHNORM = True
LAYER_WIDTH = 1024
MODEL_NAME = '16_block3_lw1024_lr1e-4'

# hyperparameters
EPOCHS = 300
BATCH_SIZE = 64
INIT_LR = 1e-4
MIN_LR = 1e-8
TRIPLET_LOSS_MARGIN = 1
REDUCE_LR_FACTOR = 0.1
REDUCE_LR_EPOCHS = 5
EARLY_STOP_EPOCHS = 30

def main():
    # prepend root directories to paths
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, f'checkpoints_{MODEL_NAME}')
    HISTORY_DIR = os.path.join(HISTORY_ROOT, f'history_{MODEL_NAME}')

    # check if directories exist
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    if not os.path.exists(HISTORY_DIR):
        os.mkdir(HISTORY_DIR)

    # initialize model
    model = CLIPMLP(clip_model=CLIP_MODEL, clip_layer=CLIP_LAYER, layer_width=LAYER_WIDTH, use_batchnorm=USE_BATCHNORM)
    model.to('cuda')

    # generate dataset
    train_paths, val_paths = get_train_val_paths(dataset_root=DATASET_ROOT, val_size=VAL_SIZE, seed=RANDOM_SEED)
    ds_train = SharpBlurImageDataset(dataset_root=DATASET_ROOT, img_paths=train_paths, img_size=IMG_SIZE)
    ds_val = SharpBlurImageDataset(dataset_root=DATASET_ROOT, img_paths=val_paths, img_size=IMG_SIZE)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)

    # optimizer
    loss_fn = nn.TripletMarginLoss(margin=TRIPLET_LOSS_MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_EPOCHS, min_lr=MIN_LR, verbose=True)

    # training
    train_losses = []
    val_losses = []
    best_val_loss = 1e10
    early_stop_count = 0

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        model.clip.eval()
        train_loss = AverageMeter()

        for sharp, blur_weak, blur_strong in tqdm(train_loader, desc='  Training'):
            # to gpu
            sharp = sharp.to('cuda')
            blur_weak = blur_weak.to('cuda')
            blur_strong = blur_strong.to('cuda')

            # get feature vectors
            f_sharp = model(sharp)
            f_blur_weak = model(blur_weak)
            f_blur_strong = model(blur_strong)

            # compute loss
            loss = loss_fn(f_sharp, f_blur_weak, f_blur_strong)
            train_loss.update(loss.item())

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_loss = AverageMeter()

        with torch.no_grad():
            for sharp, blur_weak, blur_strong in tqdm(val_loader, desc='Validating'):
                # to gpu
                sharp = sharp.to('cuda')
                blur_weak = blur_weak.to('cuda')
                blur_strong = blur_strong.to('cuda')

                # get feature vectors
                f_sharp = model(sharp)
                f_blur_weak = model(blur_weak)
                f_blur_strong = model(blur_strong)

                # compute loss
                loss = loss_fn(f_sharp, f_blur_weak, f_blur_strong)
                val_loss.update(loss.item())

        # update scheduler
        scheduler.step(val_loss.avg)

        # print status
        print(f'Epoch {epoch:03d}: train_loss: {train_loss.avg:.6f}, val_loss: {val_loss.avg:.6f}, best_val_loss: {best_val_loss:.6f}, ', end='')

        # check if validation loss improved
        improved = False
        if val_loss.avg < best_val_loss:
            improved = True
            best_val_loss = val_loss.avg
            early_stop_count = 0
            print('val_loss improved')
        else:
            early_stop_count += 1
            print(f'val_loss did not improve for {early_stop_count} epochs')
        
        # save model
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_latest.pth'))
        if improved:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'model_best.pth'))
        
        # save history
        train_losses.append(train_loss.avg)
        val_losses.append(val_loss.avg)

        with open(os.path.join(HISTORY_DIR, 'train_losses.pkl'), 'wb') as f:
            pickle.dump(train_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(HISTORY_DIR, 'val_losses.pkl'), 'wb') as f:
            pickle.dump(val_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # check early stop
        if early_stop_count >= EARLY_STOP_EPOCHS:
            print('Early stop')
            break

if __name__ == '__main__':
    main()
