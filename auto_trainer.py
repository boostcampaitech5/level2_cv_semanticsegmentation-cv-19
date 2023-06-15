import os

BASE_PATH = "/opt/ml/level2_cv_semanticsegmentation-cv-19"
CONFIG_PATH = os.path.join(BASE_PATH, "configs")
EXPERIMENT_PATH = os.path.join(BASE_PATH, "outputs")

CONFIG_QUEUE_PATH = os.path.join(CONFIG_PATH, "queue")
CONFIG_ENDS_PATH = os.path.join(CONFIG_PATH, "ends")

configs = [file for file in os.listdir(CONFIG_QUEUE_PATH) if file != ".gitkeep"]

if configs:
    print(f"current config file: {configs[0]}")

    # train.py
    os.system(f"python train.py --config {os.path.join(CONFIG_QUEUE_PATH, configs[0])}")
    os.system(f"mv {os.path.join(CONFIG_QUEUE_PATH, configs[0])} {os.path.join(CONFIG_ENDS_PATH, configs[0])}")

    # infrence.py(import json은 주석처리 해도 ruff가 못지우게 안쪽에 넣었습니다)
    # import json
    # with open(os.path.join(CONFIG_ENDS_PATH, configs[0]), "r") as exp:
    #     config = json.load(exp)
    # os.system(f"poetry run python inference.py --exp {os.path.join(EXPERIMENT_PATH, config['name'])}")
