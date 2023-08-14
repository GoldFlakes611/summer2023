import multiprocessing as mp

import cv2
import numpy as np
from scipy.stats import truncnorm
from torch.utils.data import Dataset
from torchdata.datapipes.iter import FileOpener


def preprocess_record(sample):
    return sample["steer"].item(), sample["throttle"].item(), sample["image"][0]


def process_data(sample):
    steer, throttle, image = sample
    data = np.asarray(bytearray(image), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    return steer, throttle, image


def process_image(img, aug_pixel, rangeY=(58, 282), rangeX=(208, 432), endShape=(224, 224)):
    new_rangeX = (rangeX[0] + aug_pixel, rangeX[1] + aug_pixel)
    new_img = img[rangeY[0]:rangeY[1], new_rangeX[0]:new_rangeX[1]]
    return cv2.resize(new_img, endShape)


class ImageSampler(Dataset):
    """
    This is the Image Sampler for Jetson Data
    The default sampler settings:
        1. User fisheye transform
        2. Transform to 160x90 image
        3. Augment the steering angle with truncnorm distribution
    """
    def __init__(self, tfrecords):
        self.datasets = []
        self.size = 0
        self.imgs = []
        self.steering = []
        self.throttle = []

        self.steering_factor = 208  # when steering is 1, we move 300 pixel (assumption)
        max_aug = 208
        self.max_steering_aug = max_aug / self.steering_factor
        self.scale = self.max_steering_aug / 2

        self.add_datasets(tfrecords)

    def add_datasets(self, tfrecords):
        """adds the datasets found in directories: dirs to each of their corresponding member variables

        Parameters:
        -----------
        dirs : string/list of strings
            Directories where dataset is stored"""
        self.datasets.extend(tfrecords)
        dataset = list(FileOpener(tfrecords, mode="b").load_from_tfrecord())
        with mp.Pool() as pool:
            data = pool.map(process_data, map(preprocess_record, dataset))

        self.size += len(data)
        # Transpose the data
        steering, throttle, imgs = list(zip(*data))
        self.imgs.extend(imgs)
        self.steering.extend(steering)
        self.throttle.extend(throttle)

    def prepare(self):
        """
        Run this function before sampling, and after adding all the datasets
        """
        self.steering = np.array(self.steering)
        self.throttle = np.array(self.throttle)
        self.imgs = np.stack(self.imgs)

    def process(self, img, steering, throttle):
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

        img = process_image(img, aug)
        return img, steering, throttle

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img = self.imgs[index]
        steering = self.steering[index]
        throttle = self.throttle[index]

        return *self.process(img, steering, throttle),
