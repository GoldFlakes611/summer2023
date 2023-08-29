'''
Name: trainer.py
Description: Trainer to facilitate training the model, logging, and saving
Date: 2023-08-28
Date Modified: 2023-08-28
'''
import os
import pathlib
from collections import OrderedDict

import numpy as np
import torch
from device import device
from klogs import kLogger
from metrics import angle_metric, direction_metric, loss_fn
from tqdm.auto import tqdm

import wandb

TAG = "TRAINER"
log = kLogger(TAG)


class Trainer:
    '''
    Trainer class - a class for training the model, logging, and saving
    Args:
        save_dir (str): path to save directory
        model (torch.nn.Module): model to train
        optim (torch.optim.Optimizer): optimizer to use
        turning_weight (int): weight to use for turning
        epochs (int): number of epochs to train for
        lr (float): learning rate
        bs (int): batch size
    Methods:
        load(fname)
        save(fname)
        train(sampler_train, sampler_test)
    '''
    def __init__(self, save_dir : str, model: torch.nn.Module, optim: torch.optim.Optimizer, turning_weight : int = 5, epochs : int = 200, lr : float = 1e-4, bs : int = 256):
        self.model = model
        self.optim = optim
        self.turning_weight = turning_weight
        self.epochs = epochs
        self.bs=bs
        self.lr=lr

        self.save_dir = pathlib.Path(save_dir).joinpath(model.NAME+f"_{self.epochs}_{self.lr}_{self.bs}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_log = []  # loss, angle, direction
        self.validation_log = []  # loss, angle, direction

        self.best_loss = np.inf
        self.best_angle_metric = 0
        self.best_direction_metric = 0

        self.i = 0

    def load(self) -> None:
        '''
        Loads a trainer from a file
        Args:
            fname (str): path to file
        Returns:
            None
        '''
        trainer = self.save_dir.joinpath("trainer_log.npz")
        save_model = self.save_dir.joinpath("last.pth")

        if trainer.exists() and save_model.exists():
            log.info(f"Loading trainer from {self.save_dir}")

            state = torch.load(save_model)
            self.model.load_state_dict(state["state"])
            self.optim.load_state_dict(state["optim"])

            data = np.load(trainer)
            self.i = data["i"]
            self.train_log = data["train_log"].tolist()
            self.validation_log = data["validation_log"].tolist()
            self.best_loss = data["best_loss"]
            self.best_angle_metric = data["best_angle_metric"]
            self.best_direction_metric = data["best_direction_metric"]

    def save(self) -> None:
        '''
        Saves a trainer to a file
        Args:
            fname (str): path to file
        Returns:
            None
        '''
        torch.save({
            "state": self.model.state_dict(),
            "optim": self.optim.state_dict(),
        }, self.save_dir.joinpath(f"last.pth"))

        np.savez_compressed(
            self.save_dir.joinpath("trainer_log.npz"),
            train_log=self.train_log,
            validation_log=self.validation_log,
            i=self.i,
            best_loss=self.best_loss,
            best_angle_metric=self.best_angle_metric,
            best_direction_metric=self.best_direction_metric
        )

    def train(self, sampler_train, sampler_test):
        '''
        Trains the model
        Args:
            sampler_train (torch.utils.data.DataLoader): training data
            sampler_test (torch.utils.data.DataLoader): testing data
        Returns:
            None
        '''
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
                X = img.to(device) / 256 #Starting dimensions [1,100,90,160] -> After permute [100, 1, 90, 160] (Pytorch Tensor format[Number, Channels, Height, Width])

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
                wandb.log({
                    'epoch':self.i,
                    'loss':loss,
                    'angle':ang_metric,
                    'direction':dir_metric
                })

            #Validation (Testing)
            batch_bar.set_description("Validation")
            batch_bar.reset(batches_test)
            validation_avg = torch.zeros(batches_test, 3)  # loss, angle, direction_sum
            direction_num = 0

            for j, (img, steering, throttle) in enumerate(sampler_test):
                Y = torch.stack([steering, throttle], dim=1).type(torch.float32).to(device)
                X = img.to(device) / 256

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
                wandb.log({
                    'epoch':self.i,
                    'val_loss':val_loss,
                    'val_angle':ang_metric,
                    'val_direction':dir_metric
                })

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
