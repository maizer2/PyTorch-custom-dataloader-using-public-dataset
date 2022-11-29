import torch
import torchvision.transforms as transforms
import json

from typing import Dict, Optional, Callable, List, Tuple
from PIL import Image, ImageDraw

# -------------------------------------------------- #

def val_root(root: str, train: bool) -> str:
    if root[-1] != "/":
        root += "/"

    if train:
        root += "Training/"
    else:
        root += "Validation/"

    return root 

# -------------------------------------------------- #

def read_json(json_path: str) -> Dict:
    path = open(json_path, 'r')
    return json.load(path)

# --------------------------------------------------- #

def get_item_seg_color(category_id):
    # Hat                 : 0    Hat_hidden          : 1    
    # Rsleeve             : 2    Lsleeve             : 3    
    # Torso               : 4    Top_hidden          : 5    
    # Hip                 : 6    Pants_Rsleeve       : 7    Pants_Lsleeve       : 8    Pants_hidden         : 9
    # Skirt               : 10   Skirt_hidden        : 11   
    # Shoe                : 12   Shoe_hidden         : 13

    if category_id == 0:
        return 10
    elif category_id == 1:
        return 11
    elif category_id == 2:
        return 12
    elif category_id == 3:
        return 13
    elif category_id == 4:
        return 14
    elif category_id == 5:
        return 15
    elif category_id == 6:
        return 16
    elif category_id == 7:
        return 17
    elif category_id == 8:
        return 18
    elif category_id == 9:
        return 19
    elif category_id == 10:
        return 20
    elif category_id == 11:
        return 21
    elif category_id == 12:
        return 22
    elif category_id == 13:
        return 23

# --------------------------------------------------- #

def get_model_seg_color(category_id):
    # Hair                : 0    Rsleeve             : 2    Lsleeve             : 3    Hat                 : 3
    # Torso               : 4    Top_hidden          : 5    Outer_Torse         : 6 
    # Torso               : 4    Top_hidden          : 5    Outer_Torse         : 6 
    # Torso               : 4    Top_hidden          : 5    Outer_Torse         : 6 
    # Torso               : 4    Top_hidden          : 5    Outer_Torse         : 6 
    # Torso               : 4    Top_hidden          : 5    Outer_Torse         : 6 

    if category_id == 0:
        return 10
    elif category_id == 1:
        return 11
    elif category_id == 2:
        return 12
    elif category_id == 3:
        return 13
    elif category_id == 4:
        return 14
    elif category_id == 5:
        return 15
    elif category_id == 6:
        return 16
    elif category_id == 7:
        return 17
    elif category_id == 8:
        return 18
    elif category_id == 9:
        return 19
    elif category_id == 10:
        return 20
    elif category_id == 11:
        return 21
    elif category_id == 12:
        return 22
    elif category_id == 13:
        return 23
    elif category_id == 14:
        return 24
    elif category_id == 15:
        return 25
    elif category_id == 16:
        return 26
    elif category_id == 17:
        return 27
    elif category_id == 18:
        return 28
    elif category_id == 19:
        return 29

# --------------------------------------------------- #

def get_seg_point_list(seg_list):

    point_list = []

    for i in range(len(seg_list)):
        for j in range(len(seg_list[i])):
            for k in range(len(seg_list[i][j])):

                point_list.append(seg_list[i][j][k])
    
    return point_list

# -------------------------------------------------- #

class FP_and_WI(torch.utils.data.Dataset):
    def __init__(
        self, 
        root: str, 
        train: bool = True,
        transform: Optional[Callable] = None
        ):

        # ../FP_and_WI/
        self.root = val_root(root, train)

        self.wearing_info = read_json(self.root + "labels/wearing_info.json")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.wearing_info)

    def __getitem__(self, idx):
        
        #############################################
        # Get cloth information that fits the index #
        #############################################

        wearing_info = self.wearing_info[idx]
        
        ##########################################
        # Split a information using wearing_info #
        # 1. Get model_path_info                 #
        # 2. Split Keys and Values each variable #
        ##########################################

        model_path_info = wearing_info['wearing'].split('.')[0]
        item_name_info = list(wearing_info.keys())[1:]
        item_path_info = list(wearing_info.values())[1:]

        ##########################
        #        M O D E L       #
        # 1. Get model_img       #  
        # 2. Get model_parse_img #
        # 3. Get model_pose_img  #
        # 4. Combine model       #
        ##########################

        ####################
        # Get model image  #
        # 1. Image.open    #
        # *. Image splin   #
        # 2. Transform img #
        ####################

        model_img = Image.open(self.root + f"images/Model-Image_deid/{model_path_info}.jpg").convert("RGB")
        
        ###########################################################################################################
        # Some of the images in the data set have a defect that is (1280, 720) size rather than (720, 1280) size. #
        # Image spin Using PIL.Image.transpose model                                                              #
        ###########################################################################################################

        if model_img.size == (1280, 720):
            model_img = model_img.transpose(Image.Transpose.ROTATE_270)

        if self.transform is not None:
            model_img = self.transform(model_img)

        ################################
        # Get model_parse_img          #
        # 1. Image.new                 #
        # 2. get parse info            #
        # 3. draw polygon on new Image #
        ################################

        model_parse_img = Image.new("L", (720, 1280), 0)
        draw = ImageDraw.Draw(model_parse_img)

        model_parse_info = read_json(self.root + f"labels/Model-Parse_f/{model_path_info}.json")
        for idx in range(1, len(model_parse_info) - 1):
            draw.polygon(get_seg_point_list(model_parse_info[f"region{idx}"]["segmentation"]), get_model_seg_color(model_parse_info[f"region{idx}"]["category_id"]))
        
        model_parse_img = self.transform(model_parse_img)

        ##############################
        # Get model_pose_img         #
        # 1. Image.new               #
        # 2. get pose info           #
        # 3. draw point on new Image #
        ##############################

        model_pose_img = Image.new("L", (720, 1280), 0)
        draw = ImageDraw.Draw(model_pose_img)

        model_pose_info = read_json(self.root + f"labels/Model-Pose_f/{model_path_info}.json")
        for idx in range(0, len(model_pose_info['landmarks']), 3):
            if model_pose_info['landmarks'][idx] != 0:
                draw.point((model_pose_info['landmarks'][idx], model_pose_info['landmarks'][idx + 1]), 255)

        model_pose_img = self.transform(model_pose_img)

        ############################################################
        # Get model = [model_img, model_parse_img, model_pose_img] #
        ############################################################

        model = [model_img, model_parse_img, model_pose_img]

        ##########################
        #         I T E M        #
        # 1. Get model_img       #  
        # 2. Get model_parse_img #
        # 3. Get model_pose_img  #
        # 4. Combine model       #
        ##########################

        ###################################################
        # Get item image                                  #
        # Item image is more then one item                #
        # Item image has F(ront) and B(ehind)             #
        # 1. Loop using item_path_info each value         #
        # 2. Image.open                                   #
        # 3. Transform img                                #
        # 4. Combine each item to list transform_item_img #
        ###################################################

        transform_item_F_img, transform_item_B_img = [], []

        for item in item_path_info:
            if item is not None:
                item_F_img = Image.open(self.root + f"images/Item-Image/{item}_F.jpg")
                item_B_img = Image.open(self.root + f"images/Item-Image/{item}_B.jpg")
                transform_item_F_img.append(self.transform(item_F_img))
                transform_item_B_img.append(self.transform(item_B_img))
            else:
                transform_item_F_img.append(None)
                transform_item_B_img.append(None)

        item_img = [transform_item_F_img, transform_item_B_img]
        
        #############################################################
        # Get item_info = [item_parse_info, item_pose_info]         #
        # Using item_path_info but that is more than one value      #
        # 1. Loop using item_path_info each value                   #
        # 2. Some value is None, and then reject None data          #
        # 3. Read each json file and append value to list           #
        # 4. make image using json file                             #
        # 5. Combine parse and pose to List                         #
        #############################################################

        item_F_parse, item_B_parse = [], []
        item_F_pose, item_B_pose = [], []
        

        for item in item_path_info:
            if item != None:

                ######################
                # Get F, B parse img #
                ######################

                F_parse_img = Image.new("L", (720, 1280), 0)
                F_parse_draw = ImageDraw.Draw(F_parse_img)

                F_parse_json = read_json(self.root + f"labels/Item-Parse_f/{item}_F.json")
                
                for idx in range(1, len(F_parse_json) - 3):
                    F_parse_draw.polygon(get_seg_point_list(F_parse_json[f"region{idx}"]["segmentation"]), get_item_seg_color(F_parse_json[f"region{idx}"]["category_id"]))

                item_F_parse.append(self.transform(F_parse_img))

                B_parse_img = Image.new("L", (720, 1280), 0)
                B_parse_draw = ImageDraw.Draw(B_parse_img)

                B_parse_json = read_json(self.root + f"labels/Item-Parse_f/{item}_B.json")
                for idx in range(1, len(B_parse_json) - 3):
                    B_parse_draw.polygon(get_seg_point_list(B_parse_json[f"region{idx}"]["segmentation"]), get_model_seg_color(B_parse_json[f"region{idx}"]["category_id"]))

                item_B_parse.append(self.transform(B_parse_img))

                #####################
                # Get F, B pose img #
                #####################

                F_pose_img = Image.new("L", (720, 1280), 0)
                F_pose_draw = ImageDraw.Draw(F_pose_img)

                F_pose_json = read_json(self.root + f"labels/Item-Pose_f/{item}_F.json")
                for idx in range(0, len(F_pose_json['landmarks']), 3):
                    if F_pose_json['landmarks'][idx] != 0:
                        F_pose_draw.point((F_pose_json['landmarks'][idx], F_pose_json['landmarks'][idx + 1]), 255)
                item_F_pose.append(self.transform(F_pose_img))

                B_pose_img = Image.new("L", (720, 1280), 0)
                B_pose_draw = ImageDraw.Draw(B_pose_img)

                B_pose_json = read_json(self.root + f"labels/Item-Pose_f/{item}_B.json")
                for idx in range(0, len(B_pose_json['landmarks']), 3):
                    if B_pose_json['landmarks'][idx] != 0:
                        B_pose_draw.point((B_pose_json['landmarks'][idx], B_pose_json['landmarks'][idx + 1]), 255)

                item_B_pose.append(B_pose_img)
        
        item_parse_img = [item_F_parse, item_B_parse]
        item_pose_img = [item_F_pose, item_B_pose]

        item = [item_img, item_parse_img, item_pose_img]

        return model, item

        # item = [item_img, item_parse, item_pose]
        # model = [model_img, model_parse, model_pose]
        # [model, item] -> [[model_img, model_parse, model_pose], [item_img, item_parse, item_pose]] -> [[model_img, model_parse, model_pose], [[_B, _F], [_B, _F], [_B, _F]]]
