import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def mean(img_paths):
	imgs = []
	for p in img_paths:
		img = Image.open(p)
		img = np.array(img, dtype=np.float32)
		imgs.append(img)

	mean = np.sum(np.array(imgs), axis=0) / len(imgs)
	mean = np.array(np.round(mean), dtype=np.uint8)
	out = Image.fromarray(mean, mode='RGB')
	return out

dir_root = os.path.join('GOPRO_Large_all', 'train')
dir_list = [os.path.join(dir_root, p) for p in os.listdir(dir_root)]

if not os.path.exists('dataset'):
	os.mkdir('dataset')

if not os.path.exists('dataset/sharp'):
	os.mkdir('dataset/sharp')

if not os.path.exists('dataset/blur_weak'):
	os.mkdir('dataset/blur_weak')

if not os.path.exists('dataset/blur_strong'):
	os.mkdir('dataset/blur_strong')

for d in dir_list:
	img_paths = [os.path.join(d, p) for p in os.listdir(d)]
	img_root = os.path.split(d)[-1]

	sharp_dir = os.path.join('dataset/sharp', img_root)
	if not os.path.exists(sharp_dir):
		os.mkdir(sharp_dir)

	blur_weak_dir = os.path.join('dataset/blur_weak', img_root)
	if not os.path.exists(blur_weak_dir):
		os.mkdir(blur_weak_dir)

	blur_strong_dir = os.path.join('dataset/blur_strong', img_root)
	if not os.path.exists(blur_strong_dir):
		os.mkdir(blur_strong_dir)

	for i in tqdm(range(0, len(img_paths), 5)):
		center_img = img_paths[i + 2]
		img_id = os.path.split(center_img)[-1]
		sharp = Image.open(center_img)
		blur_weak = mean(img_paths[i + 1 : i + 4])
		blur_strong = mean(img_paths[i : i + 5])
		sharp.save(os.path.join(sharp_dir, img_id))
		blur_weak.save(os.path.join(blur_weak_dir, img_id))
		blur_strong.save(os.path.join(blur_strong_dir, img_id))
