import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import wandb
from losses.base_loss import DiceCoef, create_criterion
from torch import cuda
from torch.utils.data import DataLoader
from trainer.trainer import Trainer
from utils.util import ensure_dir


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--config", type=str, default="./configs/queue/base_config.json", help="config file address")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    # Conventional args
    parser.add_argument("--name", default=config["name"], help="model save at {SM_MODEL_DIR}/{name}")
    parser.add_argument("--seed", type=int, default=config["seed"], help="random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=config["epochs"], help="number of epochs to train (default: 1)")
    parser.add_argument("--early_stop", type=int, default=config["early_stop"], help="Early stop training when 10 epochs no improvement")
    parser.add_argument("--save_interval", type=int, default=config["save_interval"], help="Model save interval")
    parser.add_argument("--log_interval", type=int, default=config["log_interval"], help="Wandb logging interva(step)")
    parser.add_argument("--is_wandb", type=str2bool, default=config["is_wandb"], help="determine whether log at Wandb or not")
    parser.add_argument("--is_debug", type=str2bool, default=config["is_debug"], help="determine whether debugging mode or not")
    parser.add_argument("--dataset", type=str, default=config["dataset"], help="dataset type (default: XRayDataset)")
    parser.add_argument(
        "--augmentation", type=str, default=config["augmentation"], help="dataset augmentation type (default: BaseAugmentation)"
    )
    # parser.add_argument("--resize", type=list, default=config["resize"], help="img resize shape (default: [512,512])")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="input batch size for training (default: 64)")

    parser.add_argument("--model", type=str, default=config["model"], help="model type (default: UNet)")
    parser.add_argument("--criterion", type=str, default=config["criterion"], help="criterion type (default: bce_with_logit)")
    parser.add_argument("--optimizer", type=str, default=config["optimizer"], help="optimizer type (default: Adam)")
    parser.add_argument("--lr_scheduler", type=str, default=config["lr_scheduler"], help="lr_scheduler type (default: StepLR)")

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=config["num_workers"])

    # Container environment
    parser.add_argument("--root_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/data"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./outputs"))
    args = parser.parse_args()

    if args.is_debug:
        args.epochs = 2
    print(args)

    return args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(r"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def main(args):
    if args.is_wandb:
        wandb.init(entity="cv-19", project="segmentation-pytorch", name=args.name, config=vars(args))

    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(args.model_dir, args.name))  # ./outputs/exp_name
    ensure_dir(save_dir)

    IMAGE_ROOT = os.path.join(args.root_dir, "train/DCM")
    LABEL_ROOT = os.path.join(args.root_dir, "train/outputs_json")

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("datasets.base_dataset"), args.dataset)  # default: XRayDataset
    train_dataset = dataset_module(IMAGE_ROOT, LABEL_ROOT, is_train=True, is_debug = args.is_debug)
    valid_dataset = dataset_module(IMAGE_ROOT, LABEL_ROOT, is_train=False, is_debug = args.is_debug)

    # -- augmentation
    transform_module = getattr(import_module("datasets.augmentation"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module

    train_dataset.set_transform(transform)
    valid_dataset.set_transform(transform)

    # -- data_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    # -- model
    model_file_name = args.model.lower() + "_custom"  # custom
    model_name = "model." + model_file_name
    model_module = getattr(import_module(model_name), args.model)  # default: UNet
    model = model_module().to(device)

    # -- loss & metric
    criterion = []
    for i in args.criterion:
        criterion.append(create_criterion(i))  # default: [bce_with_logit]

    opt_module = getattr(import_module("torch.optim"), args.optimizer["type"])  # default: AdamW
    optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), **dict(args.optimizer["args"]))

    sche_module = getattr(import_module("torch.optim.lr_scheduler"), args.lr_scheduler["type"])  # default: ReduceLROnPlateau
    scheduler = sche_module(optimizer, **dict(args.lr_scheduler["args"]))

    metrics = [DiceCoef()]

    # -- logging
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        args_dict = vars(args)
        args_dict["model_dir"] = save_dir
        args_dict["TestAugmentation"] = valid_dataset.get_transform().__str__()
        json.dump(args_dict, f, ensure_ascii=False, indent=4)

    # --train
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        save_dir,
        args=args,
        device=device,
        train_loader=train_loader,
        val_loader=valid_loader,
        lr_scheduler=scheduler,
    )

    trainer.train()


# python train.py --config ./configs/queue/base_config.json
if __name__ == "__main__":
    args = parse_args()
    main(args)
