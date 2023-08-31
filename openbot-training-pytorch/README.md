# Training Pipeline using PyTorch

## Dependencies

Explicit dependencies:

- ipykernel
- ipywidgets
- matplotlib
- numpy
- opencv-python
- pytorch
- scipy
- torchdata (optional for tfrecord support)
- torchvision
- tqdm
- ultralytics(optional for using yolo)
- wandb

## Training

Before running the script, make sure the dataset is with correct structure. We assume the dataset is in the `mydataset` folder. The structure of the dataset should be like this:

```
mydataset
├── train
│   ├── <dataset_folder_1>
│   │   ├── images
│   │   │   ├── xxxx_preview.jpeg
│   │   │   └── [* more images *]
│   │   └── sensor_data
│   │       ├── ctrlLog.txt
│   │       ├── indicatorLog.txt
│   │       ├── inferenceTime.txt
│   │       └── rgbFrames.txt
│   └── [* more dataset_folder *]
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

## Known Issues

- Training script using multiprocessing to process data, which is incompatible with GNU OpenMP. The way to work around this is to set `torch.set_num_threads(1)` for the worker process.

- NumPy prefer intel openmp while PyTorch prefer GNU OpenMP.  To ensure GNU OpenMP is used, always import PyTorch before NumPy.
