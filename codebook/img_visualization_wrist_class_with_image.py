
# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
# visualization
import matplotlib.pyplot as plt
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


# utility function
# this does not care overlap
def label2rgb(label, class_num):
    image_size = label.shape[1:] + (3,)
    image = np.zeros(image_size, dtype=np.uint8)
    # image : 2048 * 2048 * 3
    # class_label : 2048 * 2048
    class_label = label[class_num]
    image[class_label == 1] = PALETTE[i]
    return image

# 이미지 파일들이 있는 디렉토리 경로
image_dir = '../../data_all/train/img/'
label_dir = '../../data_all/train/label/'
# 결과 이미지가 저장될 폴더 경로
output_dir = '../../data_visualization_wrist_class_with_image/train/'

# 결과 이미지를 저장할 폴더가 없다면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(image_dir):
    save_path = os.path.join(output_dir, filename)
    if os.path.exists(save_path):
        continue
    file_name, ext = os.path.splitext(filename) #확장자 분리
    label_path = label_dir + file_name + '.json' #label 경로
    image =cv2.imread(os.path.join(image_dir, filename)) # 이미지 로드
    #image = image / 255.

    # process a label of shape (H, W, NC)
    label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
    label = np.zeros(label_shape, dtype=np.uint8)
    with open(label_path, "r") as f:
        annotations = json.load(f)
    annotations = annotations["annotations"]
    for ann in annotations:
        c = ann["label"]
        class_ind = CLASS2IND[c]
        points = np.array(ann["points"])
            
            # polygon to mask
        class_label = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(class_label, [points], 1)
        label[..., class_ind] = class_label

        # to tenser will be done later
    #image = image.transpose(2, 0, 1)    # make channel first
    label = label.transpose(2, 0, 1)
        
    image = torch.from_numpy(image).float()
    label = torch.from_numpy(label).float()

    # total img
    fig, ax = plt.subplots(3, 3)  # figsize=(24, 12)
    idx = [i for i in range(20, 29)]
    for i, class_id in enumerate(idx):
        output_label = torch.tensor(label2rgb(label, class_id))
        sum_tensor = np.clip((image + output_label).numpy(),0, 255).astype(np.uint8)
        ax[i // 3, i % 3].imshow(sum_tensor)

    fig.savefig(save_path)