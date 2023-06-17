import argparse
import json
import os
import pickle
from importlib import import_module

import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from datasets.base_dataset import XRayDataset, XRayInferenceDataset


def load_model(saved_model, device):
    if smp_model["use"]:
        model_module = getattr(smp, model_name)
        model = model_module(**dict(smp_model["args"])).to(device)
    else:
        model_module_name = "model." + model_name.lower() + "_custom"
        model_module = getattr(import_module(model_module_name), model_name)
        model = model_module().to(device)

    model_path = os.path.join(saved_model, args.weights)
    model.load_state_dict(torch.load(model_path, map_location=device))

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


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


def dice_coef(data, label, thr):
    mask = data >= thr
    intersection = np.sum(mask * label)
    eps = 0.0001

    return (2.0 * intersection + eps) / (np.sum(mask) + np.sum(label) + eps)


def cal_best_thr(model, data_loader, save_path):
    model = model.to(device)
    model.eval()
    class_total_data = [np.array([]) for _ in range(29)]
    class_total_label = [np.array([]) for _ in range(29)]

    # validation
    with torch.no_grad():
        for idx, (image, label) in enumerate(data_loader):
            print(f"Batch_{idx + 1}/{len(data_loader)} ...   ", end="")
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
            print("Done!")

    # optimize threshold
    print("Optimize Threshold ...")
    best_thresholds = []
    thresholds = np.linspace(3, 100, num=98)
    _, ax = plt.subplots(6, 5, figsize=(48, 40))
    ax = ax.flatten()
    for i in range(29):
        print(f"class {i +1}...   ")
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

    plt.savefig(save_path)
    print("Done!")

    return best_thresholds


def test(model, data_loader, best_thresholds):
    model = model.to(device)
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for idx, (images, image_names) in enumerate(data_loader):
            print(f"Batch_{idx + 1}/{len(data_loader)} ...   ", end="")
            images = images.to(device)
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs).detach().cpu().numpy()

            for i, thr in enumerate(best_thresholds):
                outputs[:, i, :, :] = outputs[:, i, :, :] >= thr

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{constants.IND2CLASS[c]}_{image_name}")
            print("Done!")

    return rles, filename_and_class


@torch.no_grad()
def inference(data_dir, args):
    model = load_model(exp_path, device)

    img_root = os.path.join(data_dir, "test/DCM")
    dataset = XRayInferenceDataset(img_path=img_root)
    if args.augmentation is not None:
        transform = getattr(import_module("datasets.augmentation"), args.augmentation)
        dataset.set_transform(transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    if os.path.isfile(os.path.join(exp_path, "best_thr.p")):
        with open(os.path.join(exp_path, "best_thr.p"), "rb") as file:
            best_thresholds = pickle.load(file)
    else:
        valid_img_root = os.path.join(data_dir, "train/DCM")
        valid_label_root = os.path.join(data_dir, "train/outputs_json")

        valid_dataset = XRayDataset(valid_img_root, valid_label_root, is_train=False)
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False
        )
        save_thr_path = os.path.join(exp_path, "class_best_thr.png")
        print("Calculating Best Threshold..")

        best_thresholds = cal_best_thr(model, valid_loader, save_thr_path)

        with open(os.path.join(exp_path, "best_thr.p"), "wb") as file:  # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
            pickle.dump(best_thresholds, file)

    for i, thr in enumerate(best_thresholds):
        print(f"CLASS {i+1} : {thr}")

    print("Calculating inference results..")
    rles, filename_and_class = test(model, loader, best_thresholds)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )
    save_path = os.path.join(exp_path, "output.csv")
    df.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


# python inference.py --exp Baseline
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    parser.add_argument("--exp", type=str, default="Baseline", help="exp directory address")
    parser.add_argument("--device", type=str, default=device, help="device (cuda or cpu)")
    parser.add_argument("--weights", type=str, default="best_epoch.pth", help="model weights file (default: best_epoch.pth)")
    parser.add_argument("--batch_size", type=int, default=2, help="input batch size for validing (default: 2)")
    parser.add_argument("--augmentation", type=str, default=None, help="augmentation from datasets.augmentation")

    # Container environment
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
    data_dir = args.data_dir
    device = args.device
    inference(data_dir, args)
