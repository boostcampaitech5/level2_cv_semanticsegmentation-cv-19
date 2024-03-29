import argparse
import json
import os
import pickle
import time
from importlib import import_module

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

import constants
from datasets.base_dataset import XRayInferenceDataset


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


@torch.no_grad()
def test(model, data_loader, thresholds, args):
    rles = []
    filename_and_class = []

    if args.save_logits:
        logits_path = os.path.join(exp_path, f"{args.exp}_logits")
        if not os.path.exists(logits_path):
            os.mkdir(logits_path)

    print("Calculating inference results....")
    model.eval()
    with torch.no_grad():
        for idx, (images, image_names) in enumerate(data_loader):
            print(f"Batch [{idx + 1}/{len(data_loader)}] Image Shape : {tuple(images.shape)}")
            images = images.to(device)
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs).detach().cpu().numpy()

            if args.save_logits:
                compressed_outputs = (outputs * 10000).astype(np.uint16)
                for i in range(args.batch_size):
                    np.savez_compressed(os.path.join(logits_path, os.path.basename(image_names[i])[:-4]), logits=compressed_outputs[i])

            for i, thr in enumerate(thresholds):
                outputs[:, i, :, :] = outputs[:, i, :, :] >= thr

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{constants.IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


def inference(args, thresholds):
    start = time.time()

    # 모델 로드
    model = load_model(exp_path, device)

    # 데이터 로드
    IMG_ROOT = os.path.join(args.data_dir, "test/DCM")
    dataset = XRayInferenceDataset(img_path=IMG_ROOT)
    if args.augmentation is not None:
        transform_module = getattr(import_module("datasets.augmentation"), args.augmentation)
        transform = transform_module(img_size=args.img_size, is_train=False)
        dataset.set_transform(transform)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    # 테스트 추론
    rles, filename_and_class = test(model, test_loader, thresholds, args)

    # 결과 저장
    print("Save results..")
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
    print(f"Inference time : {time.time()-start:.3f}s")


# python inference.py --exp debug_aug3 --img_size 1024 --save_logits
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    parser.add_argument("--exp", type=str, default="Baseline", help="exp directory address")
    parser.add_argument("--device", type=str, default=device, help="device (cuda or cpu)")
    parser.add_argument("--weights", type=str, default="best_epoch.pth", help="model weights file (default: best_epoch.pth)")
    parser.add_argument("--batch_size", type=int, default=4, help="input batch size for validing (default: 4)")
    parser.add_argument("--augmentation", type=str, default="BaseAugmentation", help="augmentation from datasets.augmentation")
    parser.add_argument("--base_threshold", action="store_true")
    parser.add_argument("--save_logits", action="store_true")

    # Container environment
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/data1024"))
    parser.add_argument("--img_size", type=int, default=1024)

    args = parser.parse_args()
    exp_path = os.path.join("./outputs", args.exp)

    if args.base_threshold:
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
    assert json_file is not None

    json_path = os.path.join(exp_path, json_file)
    with open(json_path, "r") as f:
        config = json.load(f)

    model_name = config["model"]
    smp_model = config["smp"]
    smp_model["use"] = str2bool(smp_model["use"])

    inference(args, thresholds)
