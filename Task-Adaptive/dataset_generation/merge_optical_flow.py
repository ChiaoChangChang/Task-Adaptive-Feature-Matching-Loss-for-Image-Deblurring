import os
import math
import numpy as np
import cv2 as cv
from tqdm import tqdm

# config
STRONG_BLUR_FRAMES = 15
STRONG_BLUR_DIST = 25

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
					qualityLevel = 0.3,
					minDistance = 7,
					blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
				maxLevel = 2,
				criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# calculate mean image
def mean(imgs):
	for img in imgs:
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		img = img.astype(np.float32)
	mean = np.sum(np.array(imgs), axis=0) / len(imgs)
	mean = np.array(np.round(mean), dtype=np.uint8)
	return mean

# calculate distance of a pair of points
def calc_dist(x):
	return math.sqrt(abs(x[0][0] - x[1][0]) ** 2 + abs(x[0][1] - x[1][1]) ** 2)

# load frames until requirements satisfied
def accumulate_frames(img_paths, start_idx, pbar):
	# variables
	frames = []
	idx = start_idx
	total_dist = 0

	# Take first frame and find corners in it
	old_frame = cv.imread(img_paths[idx])
	idx += 1
	pbar.update(1)
	frames.append(old_frame)
	old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
	p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	# load more frames
	while (total_dist < STRONG_BLUR_DIST or len(frames) < STRONG_BLUR_FRAMES) and idx < len(img_paths):
		frame = cv.imread(img_paths[idx])
		idx += 1
		pbar.update(1)
		frames.append(frame)
		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		# calculate optical flow
		p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
		# Select good points
		if p1 is not None:
			good_new = p1[st==1]
			good_old = p0[st==1]
		# calculate distance between frames
		point_pairs = list(zip(good_new, good_old))
		dist = list(map(calc_dist, point_pairs))
		avg_dist = sum(dist) / len(dist)
		total_dist += avg_dist
		# update old frame
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1, 1, 2)
	
	return frames, img_paths[start_idx:idx]

# generate sharp and two levels of blur images
def generate_and_save_image(frames, img_paths, img_root):
	sharp_dir = os.path.join('dataset/sharp', img_root)
	if not os.path.exists(sharp_dir):
		os.mkdir(sharp_dir)

	blur_weak_dir = os.path.join('dataset/blur_weak', img_root)
	if not os.path.exists(blur_weak_dir):
		os.mkdir(blur_weak_dir)

	blur_strong_dir = os.path.join('dataset/blur_strong', img_root)
	if not os.path.exists(blur_strong_dir):
		os.mkdir(blur_strong_dir)

	center_frame = len(frames) // 2
	img_id = os.path.split(img_paths[center_frame])[-1]
	sharp = frames[center_frame]
	blur_weak = mean(frames[center_frame - center_frame // 2 : center_frame + center_frame // 2])
	blur_strong = mean(frames)
	cv.imwrite(os.path.join(sharp_dir, img_id), sharp)
	cv.imwrite(os.path.join(blur_weak_dir, img_id), blur_weak)
	cv.imwrite(os.path.join(blur_strong_dir, img_id), blur_strong)

def main():
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
		idx = 0

		with tqdm(total=len(img_paths), desc=img_root) as pbar:
			while idx < len(img_paths):
				frames, paths = accumulate_frames(img_paths, idx, pbar)
				assert len(frames) == len(paths)
				if len(frames) < STRONG_BLUR_FRAMES:
					break
				generate_and_save_image(frames, paths, img_root)
				idx += len(frames)

if __name__ == '__main__':
	main()
