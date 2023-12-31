'''
Name: sampler.py
Description: Sampler for reading data from the dataset and setting up data for training
Date: 2023-08-28
Date Modified: 2023-08-28
'''
import pathlib

import mp_sharedmap
import numpy as np
import torch
from device import device
from klogs import kLogger
from openbot import list_dirs, load_labels
from scipy.stats import truncnorm
from torch.utils.data import Dataset, random_split
from torchvision import io, transforms

TAG = "SAMPLER"
log = kLogger(TAG)


def process_data(sample):
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

    return steer, throttle, image


class ImageSampler(Dataset):
    '''
    ImageSampler class - a class for sampling images from the dataset

    Args:
        dataset_path (str): path to dataset
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
    def __init__(self, dataset_path, use_cuda=True):
        if use_cuda:
            self.device = device
        else:
            self.device = torch.device('cpu')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=False),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.05, saturation=0.05)
        ])

        self.steering_factor = 208  # when steering is 1, we move 300 pixel (assumption)
        max_aug = 208
        self.max_steering_aug = max_aug / self.steering_factor
        self.scale = self.max_steering_aug / 2
        self.prepare_datasets(dataset_path)

    def load_sample_tfrecord(self, dataset_path):
        '''
        Loads a sample from a tfrecord dataset
        Args:
            dataset_path (str): path to dataset
        Returns:
            list: list of samples
        '''
        from torchdata.datapipes.iter import FileOpener
        return [
            (sample["steer"].item(), sample["throttle"].item(), sample["image"][0]) 
            for sample in FileOpener(dataset_path, mode="b").load_from_tfrecord()
        ]

    def load_sample_openbot(self, dataset_path):
        '''
        Loads a sample from an openbot dataset
        Args:
            dataset_path (str): path to dataset
        Returns:
            list: list of samples
        '''
        samples = []
        count = 0
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

        if count > 0:
            log.error(f"Found {count} missing images")

        return samples

    def load_sample(self, dataset_paths):
        '''
        Loads a sample from a generic dataset
        Args:
            dataset_paths (str): path to dataset
        Returns:
            list: list of samples
        '''
        # XXX: Some compatibility with old tfrecord datasets
        # This is not that efficient, so eventually we will  
        # put everything into a datapipe.
        samples = []
        for dataset_path in dataset_paths:
            if not pathlib.Path(dataset_path).is_dir():
                samples.extend(self.load_sample_tfrecord([dataset_path]))
            else:
                samples.extend(self.load_sample_openbot(dataset_path))

        return samples

    def prepare_datasets(self, dataset_paths):
        """adds the datasets found in directories: dirs to each of their corresponding member variables

        Parameters:
        -----------
        dirs : string/list of strings
            Directories where dataset is stored"""
        dataset = self.load_sample(dataset_paths)
        self.size = len(dataset)
        self.steering, self.throttle, self.imgs = mp_sharedmap.map(process_data, dataset)

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

    def process(self, img, steering, throttle):
        '''
        Processes an image and returns it in the correct format, applies translation
        Args:
            img (np.array): image to be processed
            steering (float): steering angle
            throttle (float): throttle angle
        Returns:
            np.array: processed image
            float: steering angle
            float: throttle
        '''
        max_steering_aug = self.max_steering_aug
        scale = self.scale

        # Translate the image randomly
        loc = steering
        l = max(-1, loc - max_steering_aug)
        r = min(1, loc + max_steering_aug)
        a, b = (l - loc) / scale, (r - loc) / scale
        r = truncnorm(a, b, loc, scale).rvs().astype('float32')
        diff = steering - r
        aug = np.floor(diff * self.steering_factor).astype('int32')
        steering = r  # The pixel to angle conversion is approximate

        img = self.process_image(img.to(self.device), aug)
        return img, steering, throttle

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img = self.imgs[index]
        steering = self.steering[index]
        throttle = self.throttle[index]

        return *self.process(img, steering, throttle),


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
            if (dataset_path / "tfrecords" / "train.tfrec").exists():
                train_datasets.append(dataset_path / "tfrecords" / "train.tfrec")
            if (dataset_path / "test").is_dir():
                test_datasets.append(dataset_path / "test")

    trainset = ImageSampler(train_datasets, use_cuda=use_cuda)
    if test_datasets:
        testset = ImageSampler(test_datasets, use_cuda=use_cuda)
    else:
        train_size = int(len(trainset) * train_test_split)
        trainset, testset = random_split(trainset, [train_size, len(trainset) - train_size])

    log.info(f"Training: {len(trainset)}, Testing {len(testset)}")
    return trainset, testset
