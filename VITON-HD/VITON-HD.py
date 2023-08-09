import os, json, random, copy

import cv2
from PIL import Image, ImageDraw
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

parse_labels_RGB = [
    [0, 0, 0],      # Background
    
    # Body
    [254, 0, 0],    # Hair
    [0, 0, 254],    # Face
    [85, 51, 0],    # Neck
    [51, 169, 220], # Left arm
    [0, 254, 254],  # Right arm
    [169, 254, 85], # Right leg
    [85, 254, 169], # Left leg
    
    # Clothes
    ## Top
    [254, 85, 0],   # T-shirt
    [0, 119, 220],  # outer
    
    ## Bottom
    [0, 85, 85],    # Pants
    [0, 128, 0],    # Skirt
    
    ## Swimsuit, Dress
    [0, 0, 85]
]
parse_labels_8bit = [
    0,      # Background
    
    # Body
    2,    # Hair
    13,    # Face
    10,    # Neck
    14, # Left arm
    15,  # Right arm
    16, # Left leg
    17, # Right leg
    
    # Clothes
    ## Top
    5,   # T-shirt
    7,  # outer
    
    ## Bottom
    9,    # Pants
    12,    # Skirt
    
    ## Swimsuit, Dress
    6
]
densepose_labels = [
    [0, 0, 0],      # Background 
    
    [251, 235, 25], # Right head
    [248, 251, 14], # Left head
    
    [20, 80, 194],  # body
    [192, 189, 96], # Left arm outside
    [145, 191, 116],# Left arm inside
    [252, 207, 46], # Left forearm outside
    [228, 191, 74], # Left forearm inside
    [216, 186, 90], # Right arm outside
    [170, 190, 105],# Right arm inside
    [35, 227, 133], # Right forearm outside
    [237, 200, 59], # Right forearm inside
    [8, 110, 221],  # Left hand
    [4, 98, 224],   # Right hand
    
    [22, 173, 184], # Left thigh(넓적다리)
    [114, 189, 130],# Left leg(종아리)
    [6, 166, 198],  # Right thigh
    [86, 187, 145], # Right leg
    
    [17, 134, 214], # Right foot
    [13, 122, 215], # Left foot
]


class VitonHDDataset(Dataset):
    
    def get_rgb_to_8bit_pixel(self, rgb_color):
        r, g, b = rgb_color
        
        return (r*6/256)*36 + (g*6/256)*6 + (b*6/256)
    
    
    def convert_format(self, file_list: list, format_A: str, format_B: str) -> list:
        new_list = []
        for file in file_list:
            assert format_A in file, f"{format_A}가 포함되지 않았습니다."
            new_list.append(file.replace(format_A, format_B))
        
        return new_list
    
    
    def get_img_path(self, root_path: str) -> dict:
        image_path = os.path.join(root_path, self.phase, "image")
        parse_path = os.path.join(root_path, self.phase, "image-parse")
        
        cloth_path = os.path.join(root_path, self.phase, "cloth")
        cloth_mask_path = os.path.join(root_path, self.phase, "cloth-mask")
        
        agnostic_path = os.path.join(root_path, self.phase, "agnostic")
        agnostic_parse_path = os.path.join(root_path, self.phase, "agnostic-parse")
        
        densepose_path = os.path.join(root_path, self.phase, "densepose")
        
        keypoint_path = os.path.join(root_path, self.phase, "openpose_img")
        keypoint_json_path = os.path.join(root_path, self.phase, "openpose_json")
        
        path = {"root_path"             : root_path,
                "image_path"            : image_path,
                "parse_path"            : parse_path,
                "cloth_path"            : cloth_path,
                "cloth_mask_path"       : cloth_mask_path,
                "agnostic_path"         : agnostic_path,
                "agnostic_parse_path"   : agnostic_parse_path,
                "densepose_path"        : densepose_path,
                "keypoint_path"         : keypoint_path,
                "keypoint_json_path"    : keypoint_json_path}
        
        return path
    
    
    def get_img_list(self, img_path: dict) -> dict:
        image_list = os.listdir(img_path["image_path"])
        parse_list = os.listdir(img_path["parse_path"])
        
        cloth_list = os.listdir(img_path["cloth_path"])
        cloth_mask_list = os.listdir(img_path["cloth_mask_path"])
        
        agnostic_list = os.listdir(img_path["agnostic_path"])
        agnostic_parse_list = os.listdir(img_path["agnostic_parse_path"])
        
        densepose_list = os.listdir(img_path["densepose_path"])
        
        keypoint_list = os.listdir(img_path["keypoint_path"])
        keypoint_json_list = os.listdir(img_path["keypoint_json_path"])
        
        _list = {"image_list"            : image_list,
                 "parse_list"            : parse_list,
                 "cloth_list"            : cloth_list,
                 "cloth_mask_list"       : cloth_mask_list,
                 "agnostic_list"         : agnostic_list,
                 "agnostic_parse_list"   : agnostic_parse_list,
                 "densepose_list"        : densepose_list,
                 "keypoint_list"         : keypoint_list,
                 "keypoint_json_list"    : keypoint_json_list}
        
        return _list
    
    
    def get_lines(self, list_path):
        lines = []
        with open(list_path, "r") as f:
            for file in f.readlines():
                lines += [file.split("\n")[0].split(" ")]
                
        return lines
    
    
    def write_pairs(self, A: list, B: list) -> str:
        txt = ""
        for idx, item in enumerate(A):
            txt += f"{item} {B[idx]}\n"
        
        return txt
    
    
    def make_pairs_list(self, list_path: str, pairs: bool):
        img_list, cloth_list = self.img_list["image_list"], self.img_list["cloth_list"]
        
        if pairs:
            txt = self.write_pairs(img_list, img_list)
        else:
            random.shuffle(cloth_list)
            txt = self.write_pairs(img_list, cloth_list)
        
        with open(list_path, "w") as f:
            f.write(txt)
    
    
    def get_pairs_list(self, list_path: str = None, pairs: bool = True) -> list:
        if list_path is None:
            list_path = os.path.join(self.img_path["root_path"], self.pairs_name)

        if os.path.isfile(list_path):
            return self.get_lines(list_path)
        else:
            self.make_pairs_list(list_path, pairs)
            return self.get_lines(list_path)
    
        
    def __init__(self, 
                 root_path: str, 
                 phase: str = "train", 
                 pairs_list_path: str = None, 
                 pairs: bool = True,
                 img_size: tuple = (1024, 768),
                 img_mean: tuple = (0.5, ),
                 img_std: tuple = (0.5, ),
                 ):
        self.phase = phase
        self.img_size = img_size
        self.pairs_name = phase + "_" + ("paired" if pairs else "unpaired") + ".txt"
        
        self.img_path : dict = self.get_img_path(root_path)
        self.img_list : dict = self.get_img_list(self.img_path)
        self.pairs_list: list = self.get_pairs_list(pairs_list_path, pairs)
        
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size), antialias=True)
        self.normalize = transforms.Normalize(img_mean, img_std)

    
    def get_numpy_array(self, image_path: os.path) -> np.ndarray:
        image = Image.open(image_path)
        image = np.array(image)
        
        return image
    
    
    def convert_numpy_to_torch(self, ndarray: np.ndarray) -> torch.Tensor:
        torch_tensor = torch.from_numpy(ndarray)
        
        if len(torch_tensor.shape) == 2:
            torch_tensor = torch_tensor.unsqueeze(-1)

        torch_tensor = torch_tensor.permute(2, 0, 1).contiguous()
        
        return torch_tensor
    
    
    def convert_bool_to_255(self, ndarray: np.ndarray) -> np.ndarray:
        return ndarray.astype(np.uint8) * 255
    
        
    def get_image(self, target_name: str) -> torch.Tensor:
        image_path = os.path.join(self.img_path["image_path"], target_name)
        image = self.get_numpy_array(image_path)
        image = self.convert_numpy_to_torch(image)
        
        return image
    
    
    def get_inshop_cloth_image(self, cloth_name: str) -> torch.Tensor:
        image_path = os.path.join(self.img_path["cloth_path"], cloth_name)
        image = self.get_numpy_array(image_path)
        image = self.convert_numpy_to_torch(image)
        
        return image
    
    
    def get_inshop_cloth_masked_image(self, cloth_name: str) -> torch.Tensor:
        image_path = os.path.join(self.img_path["cloth_mask_path"], cloth_name)
        masked_image = self.get_numpy_array(image_path)
        masked_image = (masked_image == 255).astype(np.float32)
        
        return self.resize(self.totensor(masked_image))
    
    
    def get_wearing_cloth_image(self, target_name: str) -> torch.Tensor:
        image_path = os.path.join(self.img_path["image_path"], target_name)
        image = self.get_numpy_array(image_path)
        
        parse_path = os.path.join(self.img_path["parse_path"], self.convert_format([target_name], "jpg", "png")[0])
        parse_image = self.get_numpy_array(parse_path)
        
        T_shirt , Outer, Dress = [5], [7], [6]
        
        clothes = [T_shirt, Outer, Dress]
        
        clothes_mask = np.zeros_like(parse_image)
        for item in clothes:
            clothes_mask += ((parse_image == item))
            
        inverse_clothes_mask = ~clothes_mask
        image[inverse_clothes_mask] = (0, 0, 0)
        
        wearing_cloth_image = self.convert_numpy_to_torch(image)
        wearing_cloth_mask_image = self.convert_numpy_to_torch(self.convert_bool_to_255(clothes_mask))
        
        return wearing_cloth_image, wearing_cloth_mask_image
    
    
    def get_agnostic_image(self, target_name: str) -> torch.Tensor:
        image_path = os.path.join(self.img_path["image_path"], target_name)
        agnostic_image_path = os.path.join(self.img_path["agnostic_path"], target_name)
        
        image = self.get_numpy_array(image_path)
        agnostic_image = self.get_numpy_array(agnostic_image_path)
        
        agnostic_mask = (128, 128, 128)
        mask = np.all(agnostic_image == agnostic_mask, axis=-1)
        image[mask] = (0, 0, 0)
        
        agnostic_image = self.convert_numpy_to_torch(copy.deepcopy(image))
        agnostic_mask_image = self.convert_numpy_to_torch(self.convert_bool_to_255(mask))
        
        return agnostic_image, agnostic_mask_image
    
    
    def get_keypoint_image(self, target_name):
        target_name = target_name.split(".")[0] + "_rendered.png"
        image_path = os.path.join(self.img_path["keypoint_path"], target_name)
        image = self.get_numpy_array(image_path)
        image = self.convert_numpy_to_torch(image)
        
        return image
    
    
    def get_body_image(self, target_name):
        image_path = os.path.join(self.img_path["image_path"], target_name)
        image = self.get_numpy_array(image_path)
        
        parse_path = os.path.join(self.img_path["parse_path"], self.convert_format([target_name], "jpg", "png")[0])
        parse_image = self.get_numpy_array(parse_path)
        
        Neck, Left_arm, Right_arm, Left_leg, Right_leg, T_shirt, Outer, Pants, Skirt, Dress = \
            [10], [14], [15], [16], [17], [5], [7], [9], [12], [6]
        
        clothes = [Neck, Left_arm, Right_arm, Left_leg, Right_leg, T_shirt, Outer, Pants, Skirt, Dress]
        
        body_mask = np.zeros_like(parse_image)
        for item in clothes:
            body_mask += ((parse_image == item))
            
        inverse_body_mask = ~body_mask
        image[inverse_body_mask] = (0, 0, 0)
        
        body_image = self.convert_numpy_to_torch(image)
        body_mask_image = self.convert_numpy_to_torch(self.convert_bool_to_255(body_mask))
        
        return body_image, body_mask_image
    
    
    def __getitem__(self, index):
        target_name, cloth_name = self.pairs_list[index]
        
        image = self.get_image(target_name)
        inshop_cloth_image = self.get_inshop_cloth_image(cloth_name)
        inshop_cloth_masked_image = self.get_inshop_cloth_masked_image(cloth_name)
        wearing_cloth_image, wearing_cloth_mask_image = self.get_wearing_cloth_image(target_name) 
        agnostic, agnostic_mask = self.get_agnostic_image(target_name)
        keypoint_image = self.get_keypoint_image(target_name)
        body_image, body_mask_image = self.get_body_image(target_name)
        
        return {
            "I_name":   [target_name],
            "C_name":   [cloth_name],
            "I_size":   list(self.img_size),
            "I":        image,
            "C_I":      inshop_cloth_image,
            "C_M":      inshop_cloth_masked_image,
            "W_C_I":    wearing_cloth_image,
            "W_C_M":    wearing_cloth_mask_image,
            "A_I":      agnostic,
            "A_M":      agnostic_mask,
            "K_I":      keypoint_image,
            "B_I":      body_image,
            "B_M":      body_mask_image
        }
            
    def __len__(self):
        return len(self.pairs_list)
    
    def name(self):
        return "VitonDataset"
