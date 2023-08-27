import multiprocessing
import os
import pathlib
from collections import OrderedDict
from metrics import direction_metric, angle_metric,loss_fn

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import random
from scipy import signal
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import get_model
from sampler import load_full_dataset

if torch.cuda.is_available():
    device = torch.device("cuda")

from metrics import direction_metric, angle_metric,loss_fn


class Trainer:
    def __init__(self, save_dir, model: torch.nn.Module, optim: torch.optim.Optimizer, turning_weight=5, epochs=200):
        self.model = model
        self.optim = optim
        self.turning_weight = turning_weight
        self.epochs = epochs

        self.save_dir = pathlib.Path(save_dir).joinpath(model.NAME)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_log = []  # loss, angle, direction
        self.validation_log = []  # loss, angle, direction

        self.best_loss = np.inf
        self.best_angle_metric = 0
        self.best_direction_metric = 0

        self.i = 0

    def load(self, fname):
        data = np.load(fname)
        self.i = data["i"]
        self.train_log = data["train_log"].tolist()
        self.validation_log = data["validation_log"].tolist()
        self.best_loss = data["best_loss"]
        self.best_angle_metric = data["best_angle_metric"]
        self.best_direction_metric = data["best_direction_metric"]

    def save(self, fname):
        torch.save({
            "state": self.model.state_dict(),
            "optim": self.optim.state_dict(),
        }, os.path.join(self.save_dir, f"last.pth"))

        np.savez_compressed(
            fname,
            train_log=self.train_log,
            validation_log=self.validation_log,
            i=self.i,
            best_loss=self.best_loss,
            best_angle_metric=self.best_angle_metric,
            best_direction_metric=self.best_direction_metric
        )

    def train(self, sampler_train, sampler_test):
        epochs = self.epochs
        batches_train = len(sampler_train)
        batches_test = len(sampler_test)

        epochs_bar = tqdm(total=epochs)
        epochs_bar.set_description("Epochs")
        batch_bar = tqdm(total=batches_train)

        epochs_bar.update(self.i)
        epochs_bar.refresh()
        while self.i < epochs:
            #Training
            batch_bar.set_description("Training")
            batch_bar.reset(batches_train)
            for i, (img, steering, throttle) in enumerate(sampler_train):
                Y = torch.stack([steering, throttle], dim=1).type(torch.float32).to(device)
                #Tensor Processing
                X = img.to(device).permute(0, 3, 1, 2) / 256 #Starting dimensions [1,100,90,160] -> After permute [100, 1, 90, 160] (Pytorch Tensor format[Number, Channels, Height, Width])

                #Using the model for inference
                self.optim.zero_grad()
                Y_pred = self.model(X)

                #Loss calculation and backpropogation
                loss = loss_fn(Y[:, 0], Y[:, 1], Y_pred[:, 0], Y_pred[:, 1], throttle_weight=0.2)
                loss.backward()
                self.optim.step()

                #Some extra metrics to grade performance by
                loss = loss.item()
                ang_metric = angle_metric(Y_pred, Y).item()
                dir_metric, num = direction_metric(Y_pred, Y)
                dir_metric = (dir_metric / num).item() if num > 0 else np.nan

                #Debugging/Logging
                self.train_log.append((loss, ang_metric, dir_metric))

                batch_bar.set_postfix(ordered_dict=OrderedDict(
                    Loss=f"{loss: .3f}",
                    Best_loss=f"{self.best_loss: .3f}",
                    Angle=f"{ang_metric: .3f}",
                    Best_angle=f"{self.best_angle_metric: .3f}",
                    Direction=f"{dir_metric: .3f}",
                    Best_Dir=f"{self.best_direction_metric: .3f}"
                ))
                batch_bar.update()

            #Validation (Testing)
            batch_bar.set_description("Validation")
            batch_bar.reset(batches_test)
            validation_avg = torch.zeros(batches_test, 3)  # loss, angle, direction_sum
            direction_num = 0

            for j, (img, steering, throttle) in enumerate(sampler_test):
                Y = torch.stack([steering, throttle], dim=1).type(torch.float32).to(device)
                X = img.to(device).permute(0, 3, 1, 2) / 256

                with torch.no_grad():
                    Y_pred = self.model(X)

                # Test on Validation Set
                val_loss = loss_fn(Y[:, 0], Y[:, 1], Y_pred[:, 0], Y_pred[:, 1], throttle_weight=0.2)
                ang_metric = angle_metric(Y_pred, Y)
                dir_metric, num = direction_metric(Y_pred, Y)
                validation_avg[j] = torch.stack([val_loss, ang_metric, dir_metric])
                direction_num += num

                batch_bar.set_postfix(ordered_dict=OrderedDict(
                    Loss=f"{val_loss.item(): .3f}",
                    Best_loss=f"{self.best_loss: .3f}",
                    Angle=f"{ang_metric.item(): .3f}",
                    Best_angle=f"{self.best_angle_metric: .3f}",
                    Direction=f"{dir_metric.item() / num if num > 0 else np.nan: .3f}",
                    Best_Dir=f"{self.best_direction_metric: .3f}"
                ))
                batch_bar.update()

            validation_avg = validation_avg.sum(dim=0)
            validation_avg[:2] /= batches_test
            validation_avg[2] /= direction_num

            #Debugging/Logging
            self.validation_log.append(validation_avg.tolist())

            val_loss, ang_metric, dir_metric = self.validation_log[-1]
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if self.best_loss < 0.02:
                    torch.save({
                        "state": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                    }, os.path.join(self.save_dir, f"best_loss.pth"))
            if ang_metric > self.best_angle_metric:
                self.best_angle_metric = ang_metric
                if self.best_angle_metric > 0.6:
                    torch.save({
                        "state": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                    }, os.path.join(self.save_dir, f"best_angle.pth"))
            if dir_metric > self.best_direction_metric:
                self.best_direction_metric = dir_metric
                if self.best_direction_metric > 0.8:
                    torch.save({
                        "state": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                    }, os.path.join(self.save_dir, f"best_dir.pth"))

            # Slow for large model
            # torch.save({
            #     "state": self.model.state_dict(),
            #     "optim": self.optim.state_dict(),
            # }, os.path.join(self.save_dir, f"last.pth"))

            batch_bar.refresh()
            epochs_bar.update()
            self.i += 1