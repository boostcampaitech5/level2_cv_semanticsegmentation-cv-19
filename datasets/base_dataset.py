import json
import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import Dataset

import constants
from datasets.augmentation import BaseAugmentation, TestAugmentation


# root로부터 png, json을 읽고 순서를 matching해서 반환
def collect_img_json(IMAGE_ROOT, LABEL_ROOT):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    pngs = sorted(pngs)
    jsons = sorted(jsons)

    pngs = np.array(pngs)
    jsons = np.array(jsons)

    return pngs, jsons


def split_filename(is_train, pngs, jsons, is_debug):
    # split train-valid
    # 한 폴더 안에 한 인물의 양손에 대한 `.png` 파일이 존재하기 때문에
    # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
    # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
    groups = [os.path.dirname(fname) for fname in pngs]

    # 손목 꺾인 label
    ys = [1 if (274 <= int(os.path.dirname(fname)[2:]) <= 319) or int(os.path.dirname(fname)[2:]) == 321 else 0 for fname in pngs]
    
    train_filenames = []
    train_labelnames = []
    valid_filenames = []
    valid_labelnames = []
    
    if is_debug:
        gkf = GroupKFold(n_splits=25)
        for i, (x, y) in enumerate(gkf.split(pngs, ys, groups)):
            # 0번을 validation dataset으로 사용합니다.
            if i == 0:
                valid_filenames += list(pngs[y])
                valid_labelnames += list(jsons[y])

            elif i < 5:
                train_filenames += list(pngs[y])
                train_labelnames += list(jsons[y])
    else:
        sgkf = StratifiedGroupKFold(n_splits=5)
        train, test = next(sgkf.split(pngs, ys, groups))
        ys = np.array(ys)
        train_filenames = list(pngs[train])
        train_labelnames = list(jsons[train])
        train_filenames.extend(np.array(train_filenames)[ys[train]==1]) #oversampling
        train_labelnames.extend(np.array(train_labelnames)[ys[train]==1]) #oversampling
        
        valid_filenames = list(pngs[test])
        valid_labelnames = list(jsons[test])
    
    

    return [train_filenames, train_labelnames] if is_train else [valid_filenames, valid_labelnames]


class XRayDataset(Dataset):
    def __init__(self, IMAGE_ROOT, LABEL_ROOT, transforms=BaseAugmentation, is_train=False, is_debug=False, preprocessing=None):
        self.is_train = is_train
        self.transforms = transforms
        self.IMAGE_ROOT = IMAGE_ROOT
        self.LABEL_ROOT = LABEL_ROOT
        self.preprocessing = preprocessing
        pngs, jsons = collect_img_json(IMAGE_ROOT, LABEL_ROOT)
        self.filenames, self.labelnames = split_filename(is_train, pngs, jsons, is_debug)

    def set_transform(self, transforms):
        self.transforms = transforms

    def get_transform(self):
        return self.transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0  # normalize

        label_name = self.labelnames[item]
        label_path = os.path.join(self.LABEL_ROOT, label_name)

        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(constants.CLASSES),)
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = constants.CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label
        '''
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=label)
            image, label = sample['image'], sample['mask']
        '''
        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class XRayInferenceDataset(Dataset):
    def __init__(self, img_path, transforms=TestAugmentation):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=img_path)
            for root, _dirs, files in os.walk(img_path)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        self.img_path = img_path
        self.filenames = _filenames
        self.transforms = transforms

    def set_transform(self, transforms):
        self.transforms = transforms

    def get_transform(self):
        return self.transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.img_path, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first

        image = torch.from_numpy(image).float()

        return image, image_name
