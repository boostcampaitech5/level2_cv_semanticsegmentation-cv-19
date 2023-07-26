import os
import json
from tqdm import tqdm
from PIL import Image
import os
import cv2
from tqdm import tqdm
import albumentations as A

n = int(input("저장할 이미지 크기를 입력하세요. ex)512"))
resize = (n,n)
size = (2048, 2048)

origin_root = "/opt/ml/data/train/outputs_json"
resize_root = f"/opt/ml/data{resize[0]}/train/outputs_json"

path = []
for folder in os.listdir(origin_root):
    for file in os.listdir(origin_root +"/"+ folder):
        if file[-4:].lower() == "json":
            file_path = folder + "/" + file
            path.append(file_path)

for file in tqdm(path):
    ORIGIN_ROOT = origin_root + "/" + file
    RESIZE_ROOT = resize_root + "/" + file

    folder_name = file.split("/")[0]    
    os.makedirs(resize_root + "/" + folder_name, exist_ok=True)

    with open(ORIGIN_ROOT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for anno in data["annotations"]:
        resize_point = []
        for point in anno["points"]:
            pnt = [point[0]*resize[0]//size[0],point[1]*resize[1]//size[1]]
            if pnt not in resize_point:
                resize_point.append(pnt)
        anno["points"] = resize_point

    
    
    with open( RESIZE_ROOT,'w',  encoding='utf-8') as train_writer:
            json.dump(data,
                      train_writer, indent=2, ensure_ascii=False)
    


print("Done..!")


origin_root = "/opt/ml/data/train/DCM"
resize_root = f"/opt/ml/data{resize[0]}/train/DCM"

path = []
for folder in os.listdir(origin_root):
    for file in os.listdir(origin_root +"/"+ folder):
        if file[-3:].lower() == "png":
            file_path = folder + "/" + file
            path.append(file_path)

tf = A.Resize(*resize)

for file in tqdm(path):
    ORIGIN_ROOT = origin_root + "/" + file
    RESIZE_ROOT = resize_root + "/" + file

    folder_name = file.split("/")[0]    
    os.makedirs(resize_root + "/" + folder_name, exist_ok=True)
    
    image = cv2.imread(ORIGIN_ROOT)
    image = tf(image = image)
    image = Image.fromarray(image['image'])
    image.save(RESIZE_ROOT)

print("Done..!")

origin_root = "/opt/ml/data/test/DCM"
resize_root = f"/opt/ml/data{resize[0]}/test/DCM"

path = []
for folder in os.listdir(origin_root):
    for file in os.listdir(origin_root +"/"+ folder):
        if file[-3:].lower() == "png":
            file_path = folder + "/" + file
            path.append(file_path)

tf = A.Resize(*resize)

for file in tqdm(path):
    ORIGIN_ROOT = origin_root + "/" + file
    RESIZE_ROOT = resize_root + "/" + file

    folder_name = file.split("/")[0]    
    os.makedirs(resize_root + "/" + folder_name, exist_ok=True)
    
    image = cv2.imread(ORIGIN_ROOT)
    image = tf(image = image)
    image = Image.fromarray(image['image'])
    image.save(RESIZE_ROOT)

print("Done..!")