import os
import sys
import numpy as np
import pandas as pd
import csv


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    # pixels = mask.flatten()
    pixels = np.concatenate([[0], mask, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    if str(rle) == "nan":
        return np.zeros(height * width, dtype=np.uint8)
    s = rle.split()
    flag = 0
    for i in s:
        if i.startswith("[0"):
            print(i)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img


# CSV 파일들이 저장된 폴더 경로
# 앙상블할 폴더들을 ensemble_input에 넣어주면 됩니다.
input_path = "/opt/ml/level2_cv_semanticsegmentation-cv-19/outputs/ensemble_input"
output_path = "/opt/ml/level2_cv_semanticsegmentation-cv-19/outputs/ensemble_output"
assert os.path.exists(input_path)

exp_names = os.listdir(input_path)
THRESHOLD = 0.45

# ensemble_output 폴더 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"{output_path} 폴더가 생성")

# 폴더 내의 CSV 파일들을 가져와서 읽기
exp_paths = [os.path.join(input_path, exp) for exp in exp_names]
input_file_paths = []
for exp_path in exp_paths:
    file_name = [file for file in os.listdir(exp_path) if file.endswith(".csv")][0]
    input_file_paths.append(os.path.join(exp_path, file_name))
    print(file_name)

if len(input_file_paths) == 0:
    print("input file 없음")
    sys.exit()

# CSV 파일들을 읽어 데이터프레임으로 저장
dfs = []
output_file_name = "ensemble.csv"
for file_path in input_file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)
    file_name, extension = os.path.basename(file_path).split(".")
    output_file_name = file_name + "_" + output_file_name

with open(os.path.join(output_path, output_file_name), "w", newline="") as output_file:
    writer = csv.writer(output_file)

    col_name = ["image_name", "class", "rle"]
    writer.writerow(col_name)
    col = [0, 0, 0]
    ensemble_df = pd.DataFrame()
    for i in range(len(dfs[0])):
        print(f"[{i + 1} / {len(dfs[0])}]")
        masks = [decode_rle_to_mask(str(df["rle"][i]), 2048, 2048) for df in dfs]
        ensemble_mask = np.stack(masks).mean(axis=0)  # Hard voting: 평균값 사용
        ensemble_mask = np.where(ensemble_mask >= THRESHOLD, 1, 0)  # Threshold 이상인 값은 1, 그렇지 않은 값은 0으로 변환
        for c in range(2):
            col[c] = dfs[0][col_name[c]][i]
        col[2] = encode_mask_to_rle(ensemble_mask)
        writer.writerow(col)
