# 패션 상품 및 착용 이미지 tree

## Folder reshape

### Before

```
└── 패션 상품 및 착용 이미지
    ├───Training
    │   ├───[Traing 라벨링]2021_Fashion_train_labels
    │   │   ├───Item-Parse_f
    │   │   ├───Item-Pose_f
    │   │   ├───Model-Parse_f
    │   │   ├───model-pose_f
    │   │   └───wearing_info_train.json
    │   ├───[Traing 원천]2021_Fashion_train_itemimages
    │   │   └───Item-Image 
    │   └───[Traing 원천]2021_Fashion_train_modelimages
    │       └───Model-Image_deid
    └───Validation
        ├───[Val 라벨링]2021_Fashion_val_labels
        │   ├───Item-Parse_f
        │   ├───Item-Pose_f
        │   ├───Model-Parse_f
        │   ├───Model-Pose_f
    │   │   └───wearing_info.json
        └───[Val 원천]2021_Fashion_val_images
            ├───Item-Image
            └───Model-Image_deid
```

### After

```
└── 패션 상품 및 착용 이미지
    ├───Training
    │   ├───labels
    │   │   ├───Item-Parse_f
    │   │   ├───Item-Pose_f
    │   │   ├───Model-Parse_f
    │   │   ├───model-pose_f
    │   │   └───wearing_info.json
    │   └───images
    │       ├───Item-Image 
    │       └───Model-Image_deid
    └───Validation
        ├───labels
        │   ├───Item-Parse_f
        │   ├───Item-Pose_f
        │   ├───Model-Parse_f
    │   │   ├───model-pose_f
    │   │   └───wearing_info.json
        └───images
            ├───Item-Image
            └───Model-Image_deid
```

### Detail reshape

[Traing 라벨링]2021_Fashion_train_labels -> labels

[Traing 원천]2021_Fashion_train_itemimages/Item-Image -> images

[Traing 원천]2021_Fashion_train_modelimages/Model-Image_deid -> images

[Val 라벨링]2021_Fashion_val_labels -> labels

[Val 원천]2021_Fashion_val_images -> images

wearing_info_train.json, wearing_info_val.json -> wearing_info.json

# How to use FP_and_WI

### Parameter

* root: str = "../FP_and_WI/"
* train: bool = True
    * If train is True, use the "../FP_and_WI/Training/" folder.
    * If train is False, use the "../FP_and_WI/Validation/" folder.
* transform: Optional[Callable] = None
    * torchvision.transforms

### Return shape

* Tuple[img, label]
    * img -> List[model_img, item_img]
        * model_img -> str("images/Model-Image_deid/{model_path_info}.jpg")
        * item_img -> List[item_F_img, item_B_img]
            * item_F_img -> str("images/Item-Image/{item}_F.jpg")
            * item_B_img -> str("images/Item-Image/{item}_B.jpg")
    * label -> List[model_info, item_info]
        * model_info-> List[model_parse_info, model_pose_info]
            * model_parse_info -> List["labels/Model-Parse_f/{model_path_info}.json"]
            * model_pose_info -> List["labels/Model-Pose_f/{model_path_info}.json"]
        * item_info -> List[item_name_info, List[item_parse_info, item_pose_info]]
            * item_name_info -> List[str]
            * item_parse_info -> List[item_F_parse_info, item_B_parse_info]
                * item_F_parse_info -> List["labels/Item-Parse_f/{item}_F.json"]
                * item_B_parse_info -> List["labels/Item-Parse_f/{item}_B.json"]
            * item_pose_info -> List[item_F_pose_info, item_B_pose_info]
                * item_F_pose_info -> List["labels/Item-Pose_f/{item}_F.json"]
                * item_B_pose_info -> List["labels/Item-Pose_f/{item}_B.json"]

### Link

[https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=78](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=78)