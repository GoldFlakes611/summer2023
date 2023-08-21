# Training Pipeline using PyTorch

Before running the script, make sure the dataset is with correct structure. We assume the dataset is in the `mydataset` folder. The structure of the dataset should be like this:

```
mydataset
├── train
│  ├── <dataset_folder_1>
│  │   ├── images
│  │   │   ├── xxxx_preview.jpeg
│  │   │   └── [* more images *]
│  │   └── sensor_data
│  │       ├── ctrlLog.txt
│  │       ├── indicatorLog.txt
│  │       ├── inferenceTime.txt
│  │       └── rgbFrames.txt
│  └── [* more dataset_folder *]
└── test
    └── [* same structure as train *]
```

You can use

```
tree --filelimit 10 mydataset
```

to verify the data structure.


1. Prepare the dataset

```bash
python prepare_dataset.py mydataset
```

2. Train the model

Refer to [Train.ipynb](Train.ipynb) for more details.
