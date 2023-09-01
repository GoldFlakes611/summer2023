'''
Name: seq_sampler.py
Description: Sampler for reading video data from the dataset and setting up data for training
'''
import pathlib

import mp_sharedmap
import torch
from device import device
from klogs import kLogger
from openbot import list_dirs, load_labels
from torch.utils.data import Dataset, random_split
from torchvision import io, transforms

TAG = "SEQSAMPLER"
log = kLogger(TAG)


class SeqSampler(Dataset):
    '''
    SeqSampler class - a class for sampling sequential images from the dataset

    Args:
        dataset_path (str): path to dataset
        seq_len (int): length of sequence to sample
        stride (int): stride between samples
        use_cuda (bool): whether to use cuda for image processing
            If Ture there will be ~400M * num_of_processes additional GPU memory usage.
            It's recommended to set the number of processes to round 4 if use_cuda is True.
            Use False if there is not enough GPU memory, or the CPU is not the bottleneck.

    Methods:
        prepare_datasets(tfrecords)
        load_sample(dataset_paths)
        load_sample_tfrecord(dataset_path)
        load_sample_openbot(dataset_path)
        process(img, steering, throttle)
    '''
    def __init__(self, dataset_path, seq_len=10, stride=3, use_cuda=True):
        self.seq_len = seq_len
        self.stride = stride
        if use_cuda:
            self.device = device
        else:
            self.device = torch.device('cpu')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=False),
        ])

        self.prepare_datasets(dataset_path)

    def load_sample(self, dataset_paths):
        '''
        Loads a sample from an openbot dataset
        Args:
            dataset_path (str): path to dataset
        Returns:
            list: list of samples
        '''
        samples = []
        count = 0
        for dataset_path in dataset_paths:
            for image_path, ctrl_cmd in load_labels(dataset_path, list_dirs(dataset_path)).items():
                if pathlib.Path(image_path).exists():

                    samples.append((
                        float(ctrl_cmd[1]) / 255.0,  # steer
                        float(ctrl_cmd[0]) / 255.0,  # throttle
                        image_path,  # image
                    ))
                else:
                    log.debug(f"File not found: {image_path}")
                    # XXX: do not report every missing file before we fix the frame matching problem
                    count += 1

        log.error(f"Found {count} missing images")
        return samples

    def prepare_datasets(self, dataset_paths):
        """adds the datasets found in directories: dirs to each of their corresponding member variables

        Parameters:
        -----------
        dirs : string/list of strings
            Directories where dataset is stored"""
        dataset = self.load_sample(dataset_paths)
        self.size = len(dataset) - self.seq_len * self.stride
        self.steering, self.throttle, self.imgs = mp_sharedmap.map(self.process_data, dataset)


    def process_data(self, sample):
        '''
        Processes a sample and return it in the correct formats
        Args:
            sample (tuple): tuple of (steering, throttle, image)
        Returns:
            tuple: tuple of (steering, throttle, image)
        '''
        steer, throttle, image = sample
        if isinstance(image, str):
            image = io.read_image(image, mode=io.ImageReadMode.RGB)
        else:
            data = torch.tensor(bytearray(image), dtype=torch.uint8)
            image = io.decode_image(data, mode=io.ImageReadMode.RGB)

        return steer, throttle, self.process_image(image, 0)

    def process_image(self, img, aug_pixel, rangeY=(136, 360), rangeX=(208, 432), endShape=(224, 224)):
        '''
        Processes an image and returns it in the correct format
        Args:
            img (np.array): image to be processed
            aug_pixel (int): number of pixels to augment
            rangeY (tuple): range of pixels in Y direction to crop
            rangeX (tuple): range of pixels in X direction to crop
            endShape (tuple): shape of the output image
        Returns:
            np.array: processed image
        Note:
            change the value of rangeY to(58,282) for center crop and (136,360) for bottom crop
        '''
        new_rangeX = (rangeX[0] + aug_pixel, rangeX[1] + aug_pixel)
        img = img[:, rangeY[0]:rangeY[1], new_rangeX[0]:new_rangeX[1]]
        return self.transform(img)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        stride = self.stride
        seq_len = self.seq_len

        img = self.imgs[index:index + seq_len * stride:stride]
        steering = self.steering[index + (seq_len - 1) * stride]
        throttle = self.throttle[index + (seq_len - 1) * stride]

        return img, steering, throttle


def load_full_dataset(dataset_paths, train_test_split=0.8, use_cuda=True):
    '''
    Loads a full dataset from a list of paths
    Args:
        dataset_paths (list): list of paths to datasets
        train_test_split (float): percentage of data to be used for training
        use_cuda (bool): whether to use cuda for image processing
            See ImageSampler for more details
    Returns:
        tuple: trainset, testset
    '''
    train_datasets = []
    test_datasets = []
    for dataset_path in dataset_paths:
        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.is_dir():
            train_datasets.append(dataset_path)
        else:
            if (dataset_path / "train").is_dir():
                train_datasets.append(dataset_path / "train")
            if (dataset_path / "test").is_dir():
                test_datasets.append(dataset_path / "test")

    trainset = SeqSampler(train_datasets, use_cuda=use_cuda)
    if test_datasets:
        testset = SeqSampler(test_datasets, use_cuda=use_cuda)
    else:
        train_size = int(len(trainset) * train_test_split)
        trainset, testset = random_split(trainset, [train_size, len(trainset) - train_size])

    log.info(f"Training: {len(trainset)}, Testing {len(testset)}")
    return trainset, testset
