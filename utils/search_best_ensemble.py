import os
import numpy as np
import pickle

input_path = "outputs/ensemble_input"

exp_names = os.listdir(input_path)
exp_scores = []
best_ensembles = set()

for exp in exp_names:
    dice_path = os.path.join(os.path.join(input_path, exp), "best_dicecoef.p")

    assert os.path.isfile(dice_path), "please run utils/optimize_threshold.py"

    with open(dice_path, "rb") as file:
        exp_scores.append(pickle.load(file))

exp_scores = np.array(exp_scores)

print("[클래스별 Best Model]")
for class_idx, model_idx in enumerate(np.argmax(exp_scores, axis=0)):
    best_ensembles.add(exp_names[model_idx])
    print(f"CLASS {class_idx} : {exp_names[model_idx]}")

print("\n[모델별 Dice Coefficient]")
for idx, scores in enumerate(exp_scores.mean(axis=1)):
    print(f"{exp_names[idx]} : {scores}")

print(f"Ensemble : {np.max(exp_scores, axis=0).mean()}\n")

print(f"최적 앙상블 조합 : {best_ensembles}")
