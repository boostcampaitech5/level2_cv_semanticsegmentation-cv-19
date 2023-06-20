import argparse
import json
import os
import pickle
import sys
from importlib import import_module

sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from datasets.base_dataset import XRayDataset


def load_model(exp_path, model_name, smp_model, device):
    if smp_model["use"]:
        model_module = getattr(smp, model_name)
        model = model_module(**dict(smp_model["args"])).to(device)
    else:
        model_module_name = "model." + model_name.lower() + "_custom"
        model_module = getattr(import_module(model_module_name), model_name)
        model = model_module().to(device)

    ckpt_path = os.path.join(exp_path, "best_epoch.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    return model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_coef(data, label, thr):
    mask = data >= thr
    intersection = np.sum(mask * label)
    eps = 0.0001

    return (2.0 * intersection + eps) / (np.sum(mask) + np.sum(label) + eps)


@torch.no_grad()
def cal_best_thr(model, valid_loader, exp_path, device):
    class_total_data = [np.array([]) for _ in range(29)]
    class_total_label = [np.array([]) for _ in range(29)]

    print("Validation..")
    # validation
    model.eval()
    with torch.no_grad():
        for idx, (image, label) in enumerate(valid_loader):
            print(f"Batch [{idx + 1}/{len(valid_loader)}] Image Shape : {tuple(image.shape)} Label Shape : {tuple(label.shape)}")
            image = image.to(device)

            # 추론
            outputs = model(image)
            outputs = F.interpolate(outputs, size=(label.size(-2), label.size(-1)), mode="bilinear")
            outputs = torch.sigmoid(outputs)

            # 후처리
            outputs = (outputs.cpu().numpy() * 100).astype(np.uint8)
            label = label.numpy().astype(np.uint8)

            outputs = outputs.swapaxes(1, 0).reshape(29, -1)
            label = label.swapaxes(1, 0).reshape(29, -1)

            for i in range(29):
                selected_idx = np.logical_not((outputs[i] < 3) * (label[i] == 0))
                class_total_data[i] = np.concatenate((class_total_data[i], outputs[i][selected_idx]), axis=0)
                class_total_label[i] = np.concatenate((class_total_label[i], label[i][selected_idx]), axis=0)

    # optimize threshold
    print("\nOptimize Threshold ...")
    best_thresholds, best_dicecoef = [], []
    thresholds = np.linspace(3, 100, num=98)
    _, ax = plt.subplots(6, 5, figsize=(48, 40))
    ax = ax.flatten()
    for i in range(29):
        print(f"class {i +1}...   ", end="")
        dices = []
        for threshold in thresholds:
            dices.append(dice_coef(class_total_data[i], class_total_label[i], threshold))

        best_threshold = thresholds[np.argmax(dices)]

        ax[i].plot(thresholds, dices)
        ax[i].plot(best_threshold, max(dices), "go")
        ax[i].plot([best_threshold, best_threshold], [0, max(dices)], "g--")
        ax[i].plot([0, 100], [max(dices), max(dices)], "g--")

        ax[i].set_xlabel("Threshold")
        ax[i].set_ylabel("Dice Coefficient")
        ax[i].set_title(f"class{i + 1}")

        best_thresholds.append(best_threshold / 100)
        best_dicecoef.append(max(dices))
        print("Done!")

    print("\nSave File...")
    plt.savefig(os.path.join(exp_path, "threshold_dice_graph.png"))

    with open(os.path.join(exp_path, "best_threshold.p"), "wb") as file:
        pickle.dump(best_thresholds, file)

    with open(os.path.join(exp_path, "best_dicecoef.p"), "wb") as file:
        pickle.dump(best_dicecoef, file)

    print("\n[Best Threshold]")
    for i, thr in enumerate(best_thresholds):
        print(f"CLASS {i+1} : {thr}")

    print("\n[Best Dice Coefficient]")
    for i, dice in enumerate(best_dicecoef):
        print(f"CLASS {i+1} : {dice}")

    return


# python inference.py --exp Baseline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", type=str, default="Baseline", help="exp directory address")
    parser.add_argument("--batch_size", type=int, default=2, help="input batch size for validing (default: 2)")
    parser.add_argument("--input_data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/data1024"))

    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/data"))

    args = parser.parse_args()
    exp_path = os.path.join("./outputs", args.exp)
    json_file = next((file for file in os.listdir(exp_path) if file.endswith(".json")), None)
    if json_file:
        json_path = os.path.join(exp_path, json_file)
        with open(json_path, "r") as f:
            config = json.load(f)

    model_name = config["model"]
    smp_model = config["smp"]
    smp_model["use"] = str2bool(smp_model["use"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(exp_path, model_name, smp_model, device)

    IMG_ROOT = os.path.join(args.input_data_dir, "train/DCM")
    LABEL_ROOT = os.path.join(args.data_dir, "train/outputs_json")

    valid_dataset = XRayDataset(IMG_ROOT, LABEL_ROOT, is_train=False)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    cal_best_thr(model, valid_loader, exp_path, device)
