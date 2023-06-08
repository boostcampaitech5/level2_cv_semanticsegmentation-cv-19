import argparse
import json
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor


def load_model(saved_model, device):
    model_file_name = args.model.lower()+ "_custom" #custom
    model_name = "model."+model_file_name
    model = getattr(import_module(model_name), args.model)  

    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, args):
    """ """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, device).to(device)
    model.eval()

    os.path.join(data_dir, "test/DCM")

    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    transform = Compose(
        [
            CenterCrop((360, 360)),
            # Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
        ]
    )
    dataset = TestDataset(img_paths, args.resize, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info["ans"] = preds
    save_path = os.path.join(model_dir, "output.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--exp", type=str, default="./experiment/exp", help="exp directory address")
    args = parser.parse_args()
    with open(os.path.join(args.exp, "config.json"), "r") as f:
        config = json.load(f)

    print(f"model dir: {config['model_dir']}")

    parser.add_argument("--batch_size", type=int, default=256, help="input batch size for validing (default: 1000)")
    parser.add_argument(
        "--resize", type=tuple, default=config["resize"], help="resize size for image when you trained (default: (96, 128))"
    )
    parser.add_argument("--model", type=str, default=config["model"], help="model type (default: BaseModel)")

    # Container environment
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/data"))
    parser.add_argument("--model_dir", type=str, default=config["model_dir"])

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    inference(data_dir, model_dir, args)
