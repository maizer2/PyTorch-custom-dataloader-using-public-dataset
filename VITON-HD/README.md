# VITON-HD tree
train data - 11,647 
test data - 2,032

```
├── train_paired.txt
├── train_unpaired.txt
├── test_unpaired.txt
├── test_paired.txt
├── test
└── train
    ├── agnostic
    │   ├── 00000_00.jpg
    │   └── ...
    ├── agnostic-parse
    │   ├── 00000_00.png
    │   └── ...
    ├── cloth
    │   ├── 00000_00.jpg
    │   └── ...
    ├── cloth-mask
    │   ├── 00000_00.png
    │   └── ...
    ├── densepose
    │   ├── 00000_00.jpg
    │   └── ...
    ├── image
    │   ├── 00000_00.jpg
    │   └── ...
    ├── image-parse
    │   ├── 00000_00.png
    │   └── ...
    ├── openpose_img
    │   ├── 00000_00_rendered.png
    │   └── ...
    └── openpose_json
        ├── 00000_00_keypoints.json
        └── ...
```

## How to use

```
VITON_HD(
    root_path="data/root/..",
    phase="train", # or test
    pairs_list_path="None" # pairs list path, or None
    pairs=True # True -> pairs, False -> unpairs,
    img_size=(1024,768),
    img_mean=(0.5,),
    img_std=(0.5,)
)

    
```

# Link

[https://github.com/shadow2496/VITON-HD](https://github.com/shadow2496/VITON-HD)
