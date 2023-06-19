import argparse
import json
import os
import pickle
from importlib import import_module

import constants
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets.base_dataset import XRayInferenceDataset
import segmentation_models_pytorch as smp


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


def test(model, data_loader, thresholds):
    rles = []
    filename_and_class = []

    print("Inference..")
    model.eval()
    with torch.no_grad():
        for idx, (images, image_names) in enumerate(data_loader):
            print(f"Batch_{idx + 1}/{len(data_loader)} ...   ", end="")
            images = images.to(device)
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs).detach().cpu().numpy()

            for i, thr in enumerate(thresholds):
                outputs[:, i, :, :] = outputs[:, i, :, :] >= thr

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{constants.IND2CLASS[c]}_{image_name}")
            print("Done!")

    return rles, filename_and_class


@torch.no_grad()
def inference(data_dir, args, thresholds):
    model = load_model(exp_path, device)

    img_root = os.path.join(data_dir, "test/DCM")
    dataset = XRayInferenceDataset(img_path=img_root)
    if args.augmentation != None:
        transform = getattr(import_module("datasets.augmentation"), args.augmentation)
        dataset.set_transform(transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    print("Calculating inference results..")
    rles, filename_and_class = test(model, loader, thresholds)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )
    save_path = os.path.join(exp_path, f"{args.exp}.csv")
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
    parser.add_argument("--batch_size", type=int, default=4, help="input batch size for validing (default: 4)")
    parser.add_argument("--augmentation", type=str, default=None, help="augmentation from datasets.augmentation")
    parser.add_argument("--not_use_threshold", action="store_true")

    # Container environment
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/data"))

    args = parser.parse_args()
    exp_path = os.path.join("./outputs", args.exp)

    if args.not_use_threshold:
        thresholds = [0.5 for _ in range(29)]
    else:
        threshold_path = os.path.join(exp_path, "best_threshold.p")
        assert os.path.isfile(threshold_path), "please run utils/optimize_threshold.py"
        print("Load Best Threshold...  ", end="")
        with open(threshold_path, "rb") as file:
            thresholds = pickle.load(file)
        print("Done!")

    for i, thr in enumerate(thresholds):
        print(f"CLASS {i+1} : {thr}")

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
    inference(data_dir, args, thresholds)
