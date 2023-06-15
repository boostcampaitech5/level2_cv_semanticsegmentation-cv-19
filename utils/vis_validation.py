import argparse
import os
import sys
from importlib import import_module

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

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


def collect_img(IMAGE_ROOT):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    pngs = sorted(pngs)
    pngs = np.array(pngs)

    return pngs


def validation_filename(pngs):
    groups = [os.path.dirname(fname) for fname in pngs]
    ys = [0 for _ in pngs]
    gkf = GroupKFold(n_splits=5)

    return list(pngs[gkf.split(pngs, ys, groups).__next__()[1]])


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


def error2rgb(label, pred):
    image_size = label.shape[1:] + (3,)
    error_map = np.zeros(image_size)

    for fp in label < pred:
        error_map[fp] = np.maximum(error_map[fp], np.array([255, 0, 0]))

    for fn in label > pred:
        error_map[fn] = np.maximum(error_map[fn], np.array([0, 0, 255]))

    rgb_sum = np.sum(error_map, axis=2)
    pred_sum = np.sum(pred, axis=0)

    error_map[rgb_sum == 0] = (255, 255, 255)

    image_tp = np.zeros(image_size)
    tp_idx = np.logical_and(rgb_sum == 0, pred_sum > 0)
    image_tp[tp_idx] = (0, 127, 0)

    return image_tp.astype(np.uint8), error_map.astype(np.uint8)


def draw_and_save(image, label, pred, save_path):
    tp_map, error_map = error2rgb(label, pred)
    label = label2rgb(label)
    pred = label2rgb(pred)
    ground_truth = np.clip(image + label / 3, 0, 255).astype(np.uint8)
    pred = np.clip(image + pred / 3, 0, 255).astype(np.uint8)

    confusion_map = np.clip(image + tp_map / 3, 0, 255).astype(np.uint8)
    error_mask = error_map != [255, 255, 255]
    confusion_map[error_mask] = error_map[error_mask]

    _, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()
    for i in range(4):
        ax[i].axis("off")

    ax[0].set_title("Ground Truth", fontsize=20)
    ax[0].imshow(ground_truth)

    ax[1].set_title("Prediction", fontsize=20)
    ax[1].imshow(pred)

    ax[2].set_title("Error Map", fontsize=20)
    ax[2].imshow(error_map)

    ax[3].set_title("Error Map (Projection)", fontsize=20)
    ax[3].imshow(confusion_map)
    plt.savefig(save_path)


def inference(cfg):
    dataset_module = getattr(import_module("datasets.base_dataset"), "XRayDataset")

    IMAGE_ROOT = os.path.join(cfg.data_path, "train/DCM")
    LABEL_ROOT = os.path.join(cfg.data_path, "train/outputs_json")

    pngs = collect_img(IMAGE_ROOT)
    filenames = validation_filename(pngs)

    valid_dataset = dataset_module(IMAGE_ROOT, LABEL_ROOT, is_train=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # -- settings
    assert torch.cuda.is_available(), "CUDA ERROR"
    device = torch.device("cuda")

    # -- model load
    print("Model Load...    ", end=" ")
    model_module_name = "model." + cfg.model_name.lower() + "_custom"
    model_module = getattr(import_module(model_module_name), cfg.model_name)
    model = model_module().to(device)
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location=device))
    model.eval()
    print("Done!")

    # -- inference
    with torch.no_grad():
        for i, (data, label) in enumerate(valid_loader):
            print(f"Draw {filenames[i]} ...    ", end="")
            data, label = data.to(device), label.to(device)

            pred = model(data)
            pred = torch.sigmoid(pred)
            pred = F.interpolate(pred, size=(2048, 2048), mode="bilinear")
            pred = (pred > 0.5).detach().cpu().numpy()

            label = F.interpolate(label, size=(2048, 2048), mode="bilinear")
            label = (label > 0.5).detach().cpu().numpy()

            image = 255 - cv2.imread(os.path.join(IMAGE_ROOT, filenames[i]))
            pred, label = pred[0], label[0]
            save_path = os.path.join(cfg.save_path, os.path.basename(filenames[i]))
            draw_and_save(image, label, pred, save_path)
            print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-c", "--ckpt_path", type=str, required=True)
    parser.add_argument("-s", "--save_path", type=str, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    sys.path.insert(1, os.path.abspath(".."))
    sys.path.insert(1, os.path.abspath("."))

    cfg = parse_args()
    assert os.path.isdir(cfg.data_path)
    assert os.path.isdir(cfg.save_path)
    assert os.path.isfile(cfg.ckpt_path)

    inference(cfg)
