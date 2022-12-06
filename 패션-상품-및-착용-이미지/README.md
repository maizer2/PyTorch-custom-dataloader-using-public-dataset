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
        │   └───wearing_info.json
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
        │   ├───model-pose_f
        │   └───wearing_info.json
        └───images
            ├───Item-Image
            └───Model-Image_deid
```

### DataSet Link

[https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=78](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=78)

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
* item: str = "uppor" or "pants"
* train: bool = True
    * If train is True, use the "../FP_and_WI/Training/" folder.
    * If train is False, use the "../FP_and_WI/Validation/" folder.
* transform: Optional[Callable] = None

### Return shape

* Dictionary {
            "Model" : self.transform(model_img),
            "Parse" : self.transform(model_parse_img),
            "Pose" : self.transform(model_pose_img),
            "Part" : self.transform(model_part_img)
        }
