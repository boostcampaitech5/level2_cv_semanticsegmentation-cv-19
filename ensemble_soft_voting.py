import sys
import os

sys.path.insert(0, os.getcwd())

import pickle
import constants
import numpy as np
import pandas as pd

# 모델 실험 결과 저장된 폴더 경로 (앙상블할 폴더들을 ensemble_input에 넣어주면 됩니다)
input_path = "/opt/ml/level2_cv_semanticsegmentation-cv-19/outputs/ensemble_input"
output_path = "/opt/ml/level2_cv_semanticsegmentation-cv-19/outputs/ensemble_output"
IMG_PATH = "/opt/ml/data/test/DCM"

assert os.path.exists(input_path)
assert os.path.exists(IMG_PATH)


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


def soft_voting(exp_paths, exp_names, combined_thresholds):
    # 이미지명 불러오기
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMG_PATH)
        for root, _, files in os.walk(IMG_PATH)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    image_names = np.array(sorted(pngs))

    # Soft-Voting 앙상블 수행
    rles = []
    filename_and_class = []
    for idx, image_name in enumerate(image_names):
        print(f"[{idx + 1}/{len(image_names)}] {image_name}")
        output = np.zeros((29, 2048, 2048)).astype(np.uint16)
        for exp_path, exp_name in zip(exp_paths, exp_names):
            output += np.load(os.path.join(exp_path, f"{exp_name}_logits", os.path.basename(image_name)[:-3] + "npz"))["logits"]

        for i, thr in enumerate(combined_thresholds):
            output[i, :, :] = output[i, :, :] >= thr * 10000

        for c, segm in enumerate(output):
            rle = encode_mask_to_rle(segm)
            rles.append(rle)
            filename_and_class.append(f"{constants.IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


if __name__ == "__main__":
    # ensemble_output 폴더 없으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"{output_path} 폴더가 생성")

    # Threshold 계산
    print("Combine Threshold...")
    exp_names = os.listdir(input_path)
    exp_paths = [os.path.join(input_path, exp) for exp in exp_names]
    combined_thresholds = []

    for exp in exp_names:
        thr_path = os.path.join(input_path, exp, "best_threshold.p")

        assert os.path.isfile(thr_path), "please run utils/optimize_threshold.py"

        with open(thr_path, "rb") as file:
            combined_thresholds.append(pickle.load(file))

    combined_thresholds = np.array(combined_thresholds).mean(axis=0)

    # 앙상블 수행
    print("Ensemble...")
    rles, filename_and_class = soft_voting(exp_paths, exp_names, combined_thresholds)

    # 파일 저장
    print("Save...")
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )
    save_path = os.path.join(output_path, f"sv_ensemble_from_{'_'.join(exp_names)}.csv")
    df.to_csv(save_path, index=False)
