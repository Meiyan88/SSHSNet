import numpy as np
import nibabel as nib
import pickle as pk
import os
import pandas as pd
from skimage.transform import resize

# import torch
# print(torch.__version__)
def cal_subject_level_dice(prediction, target, class_num=20):
    '''
    step1: calculate the dice of each category
    step2: remove the dice of the empty category and background, and then calculate the mean of the remaining dices.
    :param prediction: the automated segmentation result, a numpy array with shape of (h, w, d)
    :param target: the ground truth mask, a numpy array with shape of (h, w, d)
    :param class_num: total number of categories
    :return:
    '''
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    for i in range(0, class_num):
        if i not in target and i not in prediction:
            continue
        target_per_class = np.where(target == i, 1, 0).astype(np.float32)
        prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

        tp = np.sum(prediction_per_class * target_per_class)
        fp = np.sum(prediction_per_class) - tp
        fn = np.sum(target_per_class) - tp
        dsc = 2 * tp / (2 * tp + fp + fn + eps)
        dscs[i] = dsc
    dscs = np.where(dscs == -1.0, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs[1:])
    return subject_level_dice

def evaluate_demo(prediction_nii_files, target_nii_files):
    '''
    This is a demo for calculating the mean dice of all subjects.
    :param prediction_nii_files: a list which contains the .nii file paths of predicted segmentation
    :param target_nii_files: a list which contains the .nii file paths of ground truth mask
    :return:
    '''
    dscs = []
    for prediction_nii_file, target_nii_file in zip(prediction_nii_files, target_nii_files):
        prediction_nii = nib.load(prediction_nii_file)
        prediction = prediction_nii.get_data()
        target_nii = nib.load(target_nii_file)
        target = target_nii.get_data()
        prediction = resize(prediction, output_shape=target.shape, order=0, mode='constant',clip=False, preserve_range=True, anti_aliasing=True)
        dsc = cal_subject_level_dice(prediction, target, class_num=20)
        dscs.append(dsc)
    return dscs, np.mean(dscs)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='Random seed.')
parser.add_argument('--exid', type=str, default='ex0', help='ex id.')
parser.add_argument('--standerpath', type=str, default='/train/Mask',
                    help='the original mask without processing for inference')

args = parser.parse_args()

ex = args.exid
fold = args.fold
sub = 'sub' + str(fold)
savepath = './log/' + ex + '/' + sub

with open('./splitdataset.pkl', 'rb') as f:
    dataset = pk.load(f)
filename = dataset[fold][1]

prediction_nii_files = []
target_nii_files = []
for i, name in enumerate(filename):
    prediction_nii_files.append(os.path.join(savepath, name))
    # target_nii_files.append(os.path.join(os.path.join(mainpath, 'mask'), name))
    target_nii_files.append(os.path.join(args.standerpath, name.replace('Case', 'mask_case')))
ds, meands = evaluate_demo(prediction_nii_files, target_nii_files)
pd.DataFrame(data={
    'name': filename, 'dice': ds
}).to_csv(os.path.join(savepath, ex + sub + 'anti_aliasingTrue.csv'), index=False)
print(meands)

