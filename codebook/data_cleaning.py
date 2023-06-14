import os
import shutil
noise_data = ['ID363', 'ID487', 'ID523', 'ID387', 'ID543', 'ID073', 'ID288']
data_folder_name = 'data'
data_dir = f'/opt/ml/{data_folder_name}/train/DCM'
json_dir = f'/opt/ml/{data_folder_name}/train/outputs_json'
trash_dir = f'/opt/ml/{data_folder_name}/trash'
save_data_dir = f'/opt/ml/{data_folder_name}/trash/train/DCM'
save_json_dir = f'/opt/ml/{data_folder_name}/trash/train/outputs_json'
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
if not os.path.exists(save_json_dir):
    os.makedirs(save_json_dir)
for data in noise_data:
    data_path = os.path.join(data_dir,data)
    json_path = os.path.join(json_dir, data)
    print(data_path)
    if os.path.exists(data_path):
        shutil.move(data_path, save_data_dir)
        shutil.move(json_path, save_json_dir)
    else:
        print(f'{data}가 존재하지 않습니다.')