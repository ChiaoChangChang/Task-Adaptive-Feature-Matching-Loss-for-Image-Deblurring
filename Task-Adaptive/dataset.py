import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, RandomCrop, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_train_val_paths(dataset_root, val_size=0.1, seed=None):
    dataset_root = os.path.join(dataset_root, 'sharp')
    img_dirs = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root)]
    img_paths = [os.path.join(os.path.split(d)[-1], p) for d in img_dirs for p in os.listdir(d)]
    train_paths, val_paths = train_test_split(img_paths, test_size=val_size, random_state=seed)
    return train_paths, val_paths

def generate_preprocess_fn(img_size):
    return Compose([
        #Resize(img_size, interpolation=BICUBIC),
        #CenterCrop(img_size),
        RandomCrop(img_size),
        ColorJitter(brightness=0.5, hue=0.5),
        RandomHorizontalFlip(p=0.85), 
        RandomVerticalFlip(p=0.85),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) #from clip
    ])

class SharpBlurImageDataset(Dataset):
    def __init__(self, dataset_root, img_paths, img_size):
        super().__init__()
        self.dataset_root = dataset_root
        self.img_paths = img_paths
        self.img_size = img_size
        self.preprocess = generate_preprocess_fn(self.img_size)
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        suffix = self.img_paths[idx]
        sharp_path = os.path.join(self.dataset_root, 'sharp', suffix)
        blur_weak_path = os.path.join(self.dataset_root, 'blur_weak', suffix)
        blur_strong_path = os.path.join(self.dataset_root, 'blur_strong', suffix)

        sharp_img = read_image(sharp_path)
        sharp_img = sharp_img / 255.
        sharp_img = self.preprocess(sharp_img)

        blur_weak_img = read_image(blur_weak_path)
        blur_weak_img = blur_weak_img / 255.
        blur_weak_img = self.preprocess(blur_weak_img)

        blur_strong_img = read_image(blur_strong_path)
        blur_strong_img = blur_strong_img / 255.
        blur_strong_img = self.preprocess(blur_strong_img)

        return sharp_img, blur_weak_img, blur_strong_img

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train, _ = get_train_val_paths('dataset')
    dataset = SharpBlurImageDataset(dataset_root='dataset', img_paths=train, img_size=224)
    sharp, blur_weak, blur_strong = dataset.__getitem__(0)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].set_title('Sharp')
    axs[0].imshow(sharp)
    axs[1].set_title('Weak Blur')
    axs[1].imshow(blur_weak)
    axs[2].set_title('Strong Blur')
    axs[2].imshow(blur_strong)
    plt.tight_layout()
    plt.show()
