import os
from pathlib import Path

import numpy as np


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def collect_img_json(IMAGE_ROOT, LABEL_ROOT, is_train=True):
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    if is_train:
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
            for root, _dirs, files in os.walk(LABEL_ROOT)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }

        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

        pngs = sorted(pngs)
        jsons = sorted(jsons)

        pngs = np.array(pngs)
        jsons = np.array(jsons)

        return pngs, jsons
    else:
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
        pngs = sorted(pngs)
        pngs = np.array(pngs)

        return pngs


# make save folder
root_dir = "/opt/ml/data_all"
ensure_dir(root_dir)

which_img = "test"  # "test"
img_root = os.path.join(root_dir, f"{which_img}/img")
label_root = os.path.join(root_dir, f"{which_img}/label")
ensure_dir(img_root)
ensure_dir(label_root)

IMAGE_ROOT = os.path.join("/opt/ml/data", f"{which_img}/DCM")
LABEL_ROOT = os.path.join("/opt/ml/data", f"{which_img}/outputs_json")

if which_img == "train":
    pngs, jsons = collect_img_json(IMAGE_ROOT, LABEL_ROOT, is_train=True)

    for png, json in zip(pngs, jsons):
        # print(os.path.join(IMAGE_ROOT, png), os.path.join(root_dir, png.split("/")[-1]))
        os.system(f"cp  {os.path.join(IMAGE_ROOT, png)} {os.path.join(img_root, png.split('/')[-1])}")
        os.system(f"cp {os.path.join(LABEL_ROOT, json)} {os.path.join(label_root, json.split('/')[-1])}")
else:
    pngs = collect_img_json(IMAGE_ROOT, LABEL_ROOT, is_train=False)

    for png in pngs:
        os.system(f"cp  {os.path.join(IMAGE_ROOT, png)} {os.path.join(img_root, png.split('/')[-1])}")
