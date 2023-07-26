import argparse
import glob
import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]

PALETTE = [
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
    (220, 220, 0),
    (175, 116, 175),
    (250, 0, 30),
    (165, 42, 42),
    (255, 77, 255),
    (0, 226, 252),
    (182, 182, 255),
    (0, 82, 0),
    (120, 166, 157),
    (110, 76, 0),
    (174, 57, 255),
    (199, 100, 0),
    (72, 0, 118),
    (255, 179, 240),
    (0, 125, 92),
    (209, 0, 151),
    (188, 208, 182),
    (0, 220, 176),
]


def increment_path(path):
    path = Path(path)
    if not path.exists():
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(r"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2

        return f"{path}{n}"


def decode_rle_to_mask(rle, height, width):
    img = np.zeros(height * width, dtype=np.uint8)
    if type(rle) == float:
        return img.reshape(height, width)
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


def label2rgb(label):
    image_size = label.shape[1:] + (3,)
    image = np.zeros(image_size)
    labels = np.zeros(image_size)

    for i, class_label in enumerate(label):
        image[class_label == 1] += PALETTE[i]
        labels[class_label == 1] += 1

    for i in range(1, int(labels.max())):
        image[labels == i] /= i

    return image.astype(np.uint8)


def find_image_path(image_name, pngs, image_root):
    for png in pngs:
        if image_name in png:
            return os.path.join(image_root, png)


def draw_and_save(file_path, save_path, data_path):
    df = pd.read_csv(file_path)
    pngs = [
        os.path.relpath(os.path.join(root, fname), start=data_path)
        for root, _, files in os.walk(data_path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    ]

    preds = []
    for i, row in df.iterrows():
        if not preds:
            print(f"Draw {row['image_name']} ...    ", end="")
        pred = decode_rle_to_mask(df.iloc[i]["rle"], height=2048, width=2048)
        preds.append(pred)

        if len(preds) == 29:
            preds = np.stack(preds, 0)
            input_image = 255 - cv2.imread(find_image_path(row["image_name"], pngs, data_path))
            img = np.clip(input_image + label2rgb(preds) / 3, 0, 255).astype(np.uint8)
            plt.axis("off")
            plt.title("Prediction")
            plt.imshow(img)
            plt.savefig(f"{save_path}/{row['image_name']}")
            preds = []
            print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="exp directory address")
    parser.add_argument("--data_path", "-d", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/data"))
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = parse_args()

    assert os.path.isdir(cfg.data_path)

    exp_path = os.path.join("./outputs", cfg.exp)
    assert os.path.isdir(exp_path)

    file_path = next((file for file in os.listdir(exp_path) if file.endswith(".csv")), None)
    file_path = os.path.join(exp_path, file_path)
    assert os.path.isfile(file_path)

    save_path = increment_path(os.path.join(exp_path, "visualize_csv"))
    os.mkdir(save_path)

    draw_and_save(file_path, save_path, cfg.data_path)
