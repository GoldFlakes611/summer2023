import multiprocessing
from klogs import kLogger
TAG = "TRAIN"
log = kLogger(TAG)
import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import random
from scipy import signal
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

from models import get_model
from sampler import load_full_dataset
from trainer import Trainer

if torch.cuda.is_available():
    device = torch.device("cuda")

def visualize_dataset(trainset):
    plt.style.use("classic")
    fig, axes = plt.subplots(2, 6, figsize=(25, 8))
    axes = axes.flatten()
    for i in range(12):
        img, steering, throttle = trainset[random.randint(0, len(trainset))]
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"{steering: .4f}, {throttle: .4f}")
    plt.show()

def start_train(model_name, bs, epochs):
    train(model_name, bs, epochs, lr=1e-4, betas=(0.9,0.999), eps=1e-8)

def start_queue(all_models, bs, epoch):
    for model in all_models:
        train(model, bs, epoch, lr=1e-4, betas=(0.9,0.999), eps=1e-8)

def start_sweep(model_name, bs, epochs):
    run = wandb.init(project='self-driving')
    lr = wandb.config.lr
    betas = (wandb.config.b1, wandb.config.b2)
    eps = wandb.config.eps
    train(model_name, bs, epochs, lr, betas, eps)

def train(model_name, bs=256, epochs=1000, lr=1e-4, betas=(0.9,0.999), eps=1e-8):
    bs = 512
    epochs=100

    print(f"Training {model_name}_{epochs}_{lr}_{bs}")
    
    if model_name+f"_{epochs}_{lr}_{bs}" in trainers:
        trainer = trainers[model_name]
        # move the model back to device
        trainer.model = trainer.model.to(device)
        trainer.optim.load_state_dict(trainer.optim.state_dict())

    else:
        model = get_model(model_name)().to(device)
        optimizer = Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
        save_model = save_dir.joinpath(model.NAME+f"_{epochs}_{lr}_{bs}").joinpath("last.pth")
        if save_model.exists():
            print(f"Loading model from {save_model}")
            state = torch.load(save_model)
            model.load_state_dict(state["state"])
            optimizer.load_state_dict(state["optim"])

        trainer = Trainer(save_dir, model, optimizer, turning_weight=5, epochs=epochs, lr=lr, bs=bs)
        save_trainer = save_dir.joinpath(model.NAME+f"_{epochs}_{lr}_{bs}").joinpath("trainer_log.npz")
        if load_trainer and save_trainer.exists():
            print(f"Loading trainer from {save_trainer}")
            trainer.load(save_trainer)
        trainers[model_name+f"_{epochs}_{lr}_{bs}"] = trainer
        del model, optimizer


    print(trainer.save_dir)

    if trainer.i < trainer.epochs:
        try:
            sampler_train = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=multiprocessing.cpu_count())
            sampler_test = DataLoader(testset, batch_size=bs, shuffle=True, num_workers=multiprocessing.cpu_count())
            trainer.train(sampler_train, sampler_test)
        finally:
            # Move the model to CPU
            trainer.model = trainer.model.to('cpu')
            trainer.optim.load_state_dict(trainer.optim.state_dict())
            trainer.save(pathlib.Path(MODEL_SAVE_PATH).joinpath(trainer.model.NAME+f"_{epochs}_{lr}_{bs}").joinpath("trainer_log.npz"))

            try:
                # Close iterator
                sampler_test.close()
                sampler_train.close()
            except:
                pass

if __name__ == "__main__":
    #now make argparser for setting up sweep w/ wandb or regular training
    import argparse
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--task', type=str, help='Task to be done, could be [train, sweep, queue], default is train', default='train')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--visualize', type=bool, default=False, help='visualize dataset')
    parser.add_argument('model', type=str, default='resnet34', help='model name')
    parser.add_argument('data', type=str, default='dataset/outside', help='model name')
    args = parser.parse_args()
    log.info(args)

    MODEL_SAVE_PATH = "models"
    DATA = [args.data]
    log.info(f"Loading dataset from {DATA}")
    trainset, testset = load_full_dataset(DATA, train_test_split=0.8)

    if args.visualize:
        log.info("Visualize trainset")
        visualize_dataset(trainset)
        log.info("Visualize testset") 
        visualize_dataset(testset)

    trainers = {}

    save_dir = pathlib.Path(MODEL_SAVE_PATH)
    load_trainer = True

    match args.task:
        case "sweep": 
            sweep_configuration = {
                'method' : 'random',
                'name' : 'sweep',
                'metric': {'goal': 'maximize', 'name':'val_angle'},
                'parameters':
                {
                    'lr': {'max':0.01,'min':1e-6},
                    'b1': {'max':0.9, 'min':0.1},
                    'b2': {'max':0.9, 'min':0.1},
                    'eps':{'max':1e-7, 'min':1e-9}
                }
            }
            sweep_id = wandb.sweep(
              sweep=sweep_configuration, 
              project='self-driving'
              )
            wandb.agent(sweep_id, function=lambda: start_sweep(args.model, args.bs, args.epochs))
        case "queue":
            all_models = [
                "resnet34",
                "resnet18",
                "cnn"
            ]
            run = wandb.init(project='self-driving')
            start_queue(all_models, args.bs, args.epochs)
        case "train":
            run = wandb.init(project='self-driving')
            start_train(args.model, args.bs, args.epochs)
        case _:
            log.error("No action specified")



