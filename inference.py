import argparse
import json
import multiprocessing
import os
from importlib import import_module
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor
from datasets.base_dataset import XRayInferenceDataset
import torch.nn.functional as F
import constants
def load_model(saved_model, device):
    model_module_name = "model."+ model_name.lower() + "_custom"
    model_module = getattr(import_module(model_module_name), model_name)  
    model = model_module().to(device)

    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

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

def test(model, data_loader, thr=0.5):
    model = model.to(device)
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for _, (images, image_names) in enumerate(data_loader):
            images = images.cuda()
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{constants.IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class

@torch.no_grad()
def inference(data_dir, args):
    model = load_model(exp_path, device)

    img_root = os.path.join(data_dir, "test/DCM")
    transform = A.Resize(*args.resize)
    dataset = XRayInferenceDataset(img_path=img_root, transforms=transform)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    print("Calculating inference results..")
    rles, filename_and_class = test(model, loader)
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
    parser.add_argument("--weights", type=str, default="best.pth", help="model weights file (default: best.pth)")
    parser.add_argument("--batch_size", type=int, default=8, help="input batch size for validing (default: 1000)")
    parser.add_argument(
        "--resize", nargs="+", type=int, default=[512, 512], help="resize size for image when you trained (default: [512, 512])"
    )

    # Container environment
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/data"))

    args = parser.parse_args()
    exp_path = os.path.join('./outputs',args.exp)
    json_file = next((file for file in os.listdir(exp_path) if file.endswith('.json')), None)
    if json_file:
        json_path = os.path.join(exp_path, json_file)
        with open(json_path, 'r') as f:
            config = json.load(f)
    model_name = config["model"]
    data_dir = args.data_dir
    device = args.device
    inference(data_dir, args)
