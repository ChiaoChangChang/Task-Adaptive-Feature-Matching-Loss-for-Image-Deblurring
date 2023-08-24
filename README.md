# Task Adaptive Feature Matching Loss for Image Deblurring (ICIP 2023)

#### Chiao-Chang Chang, Bo-Cheng Yang, Yi-Ting Liu, Jun-Cheng Chen, I-Hong Jhuo, Yen-Yu Lin
[Paper](http://vllab.cs.nctu.edu.tw/images/paper/icip-chang23.pdf)

## Installation
Please view [MPRNet](https://github.com/swz30/MPRNet) and [HINet](https://github.com/megvii-model/HINet).
## Setting
1. For MPRNet: Replace the train.py, training.yml, losses.py, and test.py in original MPRNet with the new ones provided in this repository. 
2. For HINet: Replace the losses.py in original HINet with the new one provided in this repository and use the .yml, which you need to train or evaluate with.
3. Put Task-Adaptive files into HINet/ and MPRNet/Deblurring/ and rename Task-Adaptive to CLIP_MLP.

## Quick Run
There are shell scripts in MPRNet and HINet files for training and testing.

If you need pretrained weights for our model, download [pretrained weights](https://drive.google.com/drive/folders/1KtzEGdKiag2xLWQABh8CteCcM_TUd0Ad?usp=sharing).

## Contact
If you have any question, please contact chiaochangchang@gmail.com.