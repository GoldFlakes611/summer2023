'''
Name: sampler.py
Description: Sampler for reading data from the dataset and setting up data for training
Date: 2023-08-28
Date Modified: 2023-08-28
'''
import multiprocessing as mp
import pathlib

import cv2
import numpy as np
import torch
from klogs import kLogger
from openbot import list_dirs, load_labels
from scipy.stats import truncnorm
from torch.utils.data import Dataset, random_split
from torchvision import transforms

TAG = "SAMPLER"
log = kLogger(TAG)

if torch.cuda.is_available():
    device = torch.device("cuda")


def process_data(sample):
    '''
    Processes a sample and return it in the correct formats
    Args:
        sample (tuple): tuple of (steering, throttle, image)
    Returns:
        tuple: tuple of (steering, throttle, image)
    '''
    steer, throttle, image = sample
    data = np.asarray(bytearray(image), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    return steer, throttle, image


class ImageSampler(Dataset):
    '''
    ImageSampler class - a class for sampling images from the dataset

    Args:
        dataset_path (str): path to dataset
    
    Methods:
        prepare_datasets(tfrecords)
        load_sample(dataset_paths)
        load_sample_tfrecord(dataset_path)
        load_sample_openbot(dataset_path)
        process(img, steering, throttle)
    '''
    def __init__(self, dataset_path):
        self.datasets = []
        self.size = 0
        self.imgs = []
        self.steering = []
        self.throttle = []

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
        for image_path, ctrl_cmd in load_labels(dataset_path, list_dirs(dataset_path)).items():
            try:
                with open(image_path, "rb") as f:
                    image = f.read()
                samples.append((
                    float(ctrl_cmd[1]) / 255.0,  # steer
                    float(ctrl_cmd[0]) / 255.0,  # throttle
                    image,  # image
                ))
            except FileNotFoundError:
                log.error(f"File not found: {image_path}")

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

    def prepare_datasets(self, tfrecords):
        """adds the datasets found in directories: dirs to each of their corresponding member variables

        Parameters:
        -----------
        dirs : string/list of strings
            Directories where dataset is stored"""
        self.datasets.extend(tfrecords)
        dataset = self.load_sample(tfrecords)
        with mp.Pool() as pool:
            data = pool.map(process_data, dataset)

        self.size += len(data)
        # Transpose the data
        steering, throttle, imgs = list(zip(*data))
        self.imgs.extend(imgs)
        self.steering.extend(steering)
        self.throttle.extend(throttle)
        self._prepare()

    def _prepare(self):
        """
        Run this function before sampling, and after adding all the datasets
        """
        self.steering = np.array(self.steering)
        self.throttle = np.array(self.throttle)
        self.imgs = torch.from_numpy(np.stack(self.imgs))

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
        img = img[rangeY[0]:rangeY[1], new_rangeX[0]:new_rangeX[1]]
        return self.transform(img.permute(2, 0, 1))

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

        img = self.process_image(img.to(device), aug)
        return img, steering, throttle

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img = self.imgs[index]
        steering = self.steering[index]
        throttle = self.throttle[index]

        return *self.process(img, steering, throttle),


def load_full_dataset(dataset_paths, train_test_split=0.8):
    '''
    Loads a full dataset from a list of paths
    Args:
        dataset_paths (list): list of paths to datasets
        train_test_split (float): percentage of data to be used for training
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

    trainset = ImageSampler(train_datasets)
    if test_datasets:
        testset = ImageSampler(test_datasets)
    else:
        train_size = int(len(trainset) * train_test_split)
        trainset, testset = random_split(trainset, [train_size, len(trainset) - train_size])

    log.info(f"Training: {len(trainset)}, Testing {len(testset)}")
    return trainset, testset
