import os
import pickle
import numpy as np

# 모델 실험 결과 저장된 폴더 경로 (앙상블할 폴더들을 ensemble_input에 넣어주면 됩니다)
input_path = "/opt/ml/level2_cv_semanticsegmentation-cv-19/outputs/ensemble_input"
output_path = "/opt/ml/level2_cv_semanticsegmentation-cv-19/outputs/ensemble_output"
assert os.path.exists(input_path)

# ensemble_output 폴더 없으면 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"{output_path} 폴더가 생성")

exp_names = os.listdir(input_path)
exp_scores = []

# 클래스별 Best Model 계산
print("Calculate Class Best Model...")
for exp in exp_names:
    dice_path = os.path.join(input_path, exp, "best_dicecoef.p")

    assert os.path.isfile(dice_path), "please run utils/optimize_threshold.py"

    with open(dice_path, "rb") as file:
        exp_scores.append(pickle.load(file))

class_best_model = np.argmax(np.array(exp_scores), axis=0)

# Output File 생성
output_file_path = os.path.join(output_path, f"cw_ensemble_from_{'_'.join(exp_names)}.csv")
output_file = open(output_file_path, "w")

# Input File 로드
print("Load Input CSV File...")
inputs_files_data = []
data_length = None
for i, exp in enumerate(exp_names):
    if i not in class_best_model:
        inputs_files_data.append(None)
    else:
        exp_path = os.path.join(input_path, exp_names[i])
        input_file_name = [f for f in os.listdir(exp_path) if f.endswith(".csv")][0]

        with open(os.path.join(exp_path, input_file_name), "r") as file:
            data = file.readlines()
            inputs_files_data.append(data)
            data_length = len(data)
            header_line = data[0]

# Ensemble
print("Ensemble...")
for i in range(data_length):
    if i == 0:
        output_file.write(header_line)
    else:
        assert inputs_files_data[class_best_model[i % 29 - 1]] is not None
        output_file.write(inputs_files_data[class_best_model[i % 29 - 1]][i])

print("Ensemble Done!")
print(f"Save Path : {output_file_path}")
