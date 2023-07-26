import argparse
import glob
import json
import os
import re
import sys
from importlib import import_module
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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


def load_model(model_name, ckpt_path, smp_model, device):
    if smp_model["use"]:
        model_module = getattr(smp, model_name)
        model = model_module(**dict(smp_model["args"])).to(device)
    else:
        model_module_name = "model." + model_name.lower() + "_custom"
        model_module = getattr(import_module(model_module_name), model_name)
        model = model_module().to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    return model


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


def inference(cfg, exp_path, model_name, ckpt_path, smp_model):
    dataset_module = getattr(import_module("datasets.base_dataset"), "XRayDataset")

    IMAGE_ROOT = os.path.join(cfg.data_path, "train/DCM")
    LABEL_ROOT = os.path.join(cfg.data_path, "train/outputs_json")

    pngs = collect_img(IMAGE_ROOT)
    filenames = validation_filename(pngs)

    valid_dataset = dataset_module(IMAGE_ROOT, LABEL_ROOT, is_train=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # -- settings
    assert torch.cuda.is_available(), "CUDA ERROR"
    device = torch.device("cuda")

    # -- model load
    print("Model Load...    ", end=" ")
    model = load_model(model_name, ckpt_path, smp_model, device)
    print("Done!")

    # -- save path
    save_path = increment_path(os.path.join(exp_path, "visualize_val"))
    os.mkdir(save_path)

    # -- inference
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(valid_loader):
            data = data.to(device)

            pred = model(data)
            pred = torch.sigmoid(pred)
            pred = F.interpolate(pred, size=(label.size(-2), label.size(-1)), mode="bilinear")
            pred = (pred > 0.5).detach().cpu().numpy()

            label = label.numpy()
            for idx in range(cfg.batch_size):
                print(f"Draw {filenames[i * cfg.batch_size + idx]} ...    ", end="")
                file_path = os.path.join(save_path, os.path.basename(filenames[i * cfg.batch_size + idx]))
                image = 255 - cv2.imread(os.path.join(IMAGE_ROOT, filenames[i * cfg.batch_size + idx]))
                draw_and_save(image, label[idx], pred[idx], file_path)
                print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="exp directory address")
    parser.add_argument("--data_path", "-d", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/data"))
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    sys.path.insert(1, os.path.abspath(".."))
    sys.path.insert(1, os.path.abspath("."))

    cfg = parse_args()
    exp_path = os.path.join("./outputs", cfg.exp)
    assert os.path.isdir(exp_path)

    json_file = next((file for file in os.listdir(exp_path) if file.endswith(".json")), None)
    if json_file:
        json_path = os.path.join(exp_path, json_file)
        with open(json_path, "r") as f:
            config = json.load(f)
    model_name = config["model"]
    ckpt_path = os.path.join(exp_path, "best_epoch.pth")
    smp_model = config["smp"]
    smp_model["use"] = str2bool(smp_model["use"])

    inference(cfg, exp_path, model_name, ckpt_path, smp_model)
