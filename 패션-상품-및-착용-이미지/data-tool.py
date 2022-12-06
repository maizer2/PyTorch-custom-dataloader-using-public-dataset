import os
import json
import argparse
from typing import Dict
from tqdm import tqdm

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--root", default="../FP_and_WI", help="FP_and_WI Directory")
    opt = parser.parse_args()
    return opt
    
def get_json(json_path: str) -> Dict:

    path = open(json_path, 'r')
    return json.load(path)

def data_tool(data_root:str, json_root:str):

    with open(json_root) as f:
        wearing_info = get_json(json_root)

    for idx in range(len(wearing_info)):
        if len(wearing_info[idx]["wearing"].split(".")) > 2:
            wrong_name = wearing_info[idx]["wearing"]
            change_name = wearing_info[idx]["wearing"].split(".")
            change_name = change_name[0] + change_name[1] + "." + change_name[2]

            # json 파일 수정
            wearing_info[idx]["wearing"] = change_name
            
            
            # 파일 이름 수정
            os.rename(data_root + "/images/Model-Image_deid/" + wrong_name, data_root + "/images/Model-Image_deid/" + change_name)
            os.rename(data_root + "/labels/Model-Parse_f/" + wrong_name.split("jpg")[0] + "json", data_root + "/labels/Model-Parse_f/" + change_name.split("jpg")[0] + "json")
            os.rename(data_root + "/labels/Model-Pose_f/" + wrong_name.split("jpg")[0] + "json", data_root + "/labels/Model-Pose_f/" + change_name.split("jpg")[0] + "json")
        elif wearing_info[idx]["wearing"] == "1119_1119_720_D_D022_458_458_D022_000.jpg":
            del wearing_info[idx]

            # 파일 이름 수정
            os.remove(data_root+"labels/Model-Parse_f/1119_1119_720_D_D022_458_458_D022_000.json")
            os.remove(data_root+"labels/Model-Posse_f/1119_1119_720_D_D022_458_458_D022_000.json")
            break

    with open(json_root, 'w') as file:
        json.dump(wearing_info, file)

if __name__ == "__main__":
    opt = get_opt()

    data_tool(
        opt.root + "/Training/", 
        opt.root + "labels/wearing_info.json"
        )
