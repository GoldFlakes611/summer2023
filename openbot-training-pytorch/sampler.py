import multiprocessing as mp
import pathlib

import cv2
import numpy as np
from openbot import list_dirs, load_labels
from scipy.stats import truncnorm
from torch.utils.data import Dataset


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
    def __init__(self, dataset_path):
        self.datasets = []
        self.size = 0
        self.imgs = []
        self.steering = []
        self.throttle = []

        self.steering_factor = 208  # when steering is 1, we move 300 pixel (assumption)
        max_aug = 208
        self.max_steering_aug = max_aug / self.steering_factor
        self.scale = self.max_steering_aug / 2
        self.add_datasets(dataset_path)

    def load_sample_tfrecord(self, dataset_path):
        from torchdata.datapipes.iter import FileOpener
        return [
            (sample["steer"].item(), sample["throttle"].item(), sample["image"][0]) 
            for sample in FileOpener(dataset_path, mode="b").load_from_tfrecord()
        ]

    def load_sample_openbot(self, dataset_path):
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
                print(f"File not found: {image_path}")

        return samples

    def load_sample(self, dataset_paths):
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

    def add_datasets(self, tfrecords):
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
