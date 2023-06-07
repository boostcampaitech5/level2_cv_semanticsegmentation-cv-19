import json
import os

import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

import constants


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


def split_filename(is_train, pngs, jsons):
    # split train-valid
    # 한 폴더 안에 한 인물의 양손에 대한 `.png` 파일이 존재하기 때문에
    # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
    # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
    groups = [os.path.dirname(fname) for fname in pngs]

    # dummy label
    ys = [0 for fname in pngs]

    # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
    # 5으로 설정하여 GroupKFold를 수행합니다.
    gkf = GroupKFold(n_splits=5)

    train_filenames = []
    train_labelnames = []
    valid_filenames = []
    valid_labelnames = []
    for i, (x, y) in enumerate(gkf.split(pngs, ys, groups)):
        # 0번을 validation dataset으로 사용합니다.
        if i == 0:
            valid_filenames += list(pngs[y])
            valid_labelnames += list(jsons[y])

        else:
            train_filenames += list(pngs[y])
            train_labelnames += list(jsons[y])

    return [train_filenames, train_labelnames] if is_train else [valid_filenames, valid_labelnames]

# Normalize를 할 경우, cv2.imread에서 에러 발생, OpenCV depth of image unsupported (CV_64F)
BaseAugmentation = A.Compose(
    [
        A.Resize(512, 512),
        # A.ColorJitter(0.5, 0.5, 0.5, 0.25),
        # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


class XRayDataset(Dataset):
    def __init__(self, IMAGE_ROOT, LABEL_ROOT, transforms=BaseAugmentation, is_train=False):
        self.is_train = is_train
        self.transforms = transforms
        self.IMAGE_ROOT = IMAGE_ROOT
        self.LABEL_ROOT = LABEL_ROOT

        pngs, jsons = collect_img_json(IMAGE_ROOT, LABEL_ROOT)
        self.filenames, self.labelnames = split_filename(is_train, pngs, jsons)

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

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

    # utility function, this does not care overlap
    @staticmethod
    def label2rgb(label):
        image_size = label.shape[1:] + (3,)
        image = np.zeros(image_size, dtype=np.uint8)

        for i, class_label in enumerate(label):
            image[class_label == 1] = constants.PALETTE[i]

        return image
