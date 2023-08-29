'''
Name: train.py
Description: Train utility, can be used instead of Train.ipynb 
Date: 2023-08-28
Date Modified: 2023-08-28
'''
import pathlib

import torch
from dataloader import FastDataLoader
from device import device
from klogs import kLogger
from matplotlib import pyplot as plt
from models import get_model
from numpy import random
from sampler import load_full_dataset
from torch.optim import Adam
from trainer import Trainer

import wandb

TAG = "TRAIN"
log = kLogger(TAG)


def visualize_dataset(trainset : torch.utils.data.Dataset) -> None:
    '''
    Visualizes the dataset, prints a random sample of 12 images
    Args:
        trainset (torch.utils.data.Dataset): dataset to visualize
    Returns:
        None
    '''
    plt.style.use("classic")
    fig, axes = plt.subplots(2, 6, figsize=(25, 8))
    axes = axes.flatten()
    for i in range(12):
        img, steering, throttle = trainset[random.randint(0, len(trainset))]
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"{steering: .4f}, {throttle: .4f}")
    plt.show()

def start_train(model_name : str, bs : int, epochs : int) -> None:
    '''
    Wrapper function for training single model
    Args:
        model_name (str): name of model to train
        bs (int): batch size
        epochs (int): number of epochs to train for
    Returns:
        None
    '''
    train(model_name, bs, epochs, lr=1e-4, betas=(0.9,0.999), eps=1e-8)

def start_queue(all_models : list, bs : int, epoch : int) -> None:
    '''
    Wrapper function to train multiple models in sequence 
    Args:
        all_models (list): list of models to train
        bs (int): batch size
        epochs (int): number of epochs to train for
    Returns:
        None
    '''
    for model in all_models:
        train(model, bs, epoch, lr=1e-4, betas=(0.9,0.999), eps=1e-8)

def start_sweep(model_name : str, bs : int, epochs : int) -> None:
    '''
    Wrapper function for hyper parameter search of a single model 
    Args:
        model_name (str): name of model to train
        bs (int): batch size
        epochs (int): number of epochs to train for
    Returns:
        None
    '''
    run = wandb.init(project='self-driving')
    lr = wandb.config.lr
    betas = (wandb.config.b1, wandb.config.b2)
    eps = wandb.config.eps
    train(model_name, bs, epochs, lr, betas, eps)

def train(model_name : str, bs : int = 256, epochs : int = 1000, lr : float = 1e-4, betas : tuple = (0.9,0.999), eps : float = 1e-8) -> None:
    '''
    Main training function - wrapped by options 
    Args:
        model_name (str): name of model to train
        bs (int): batch size
        epochs (int): number of epochs to train for
        lr (float): learning rate
        betas (tuple): betas for Adam optimizer
        eps (float): epsilon for Adam optimizer
    Returns:
        None
    '''
    log.info(f"Training {model_name}_{epochs}_{lr}_{bs}")
    
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
            log.info(f"Loading model from {save_model}")
            state = torch.load(save_model)
            model.load_state_dict(state["state"])
            optimizer.load_state_dict(state["optim"])

        trainer = Trainer(save_dir, model, optimizer, turning_weight=5, epochs=epochs, lr=lr, bs=bs)
        if load_trainer:
            log.info(f"Loading trainer")
            trainer.load()

        trainers[model_name+f"_{epochs}_{lr}_{bs}"] = trainer
        del model, optimizer


    if trainer.i < trainer.epochs:
        try:
            sampler_train = FastDataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)
            sampler_test = FastDataLoader(testset, batch_size=bs, shuffle=True, num_workers=4)
            trainer.train(sampler_train, sampler_test)
        finally:
            # Move the model to CPU
            trainer.model = trainer.model.to('cpu')
            trainer.optim.load_state_dict(trainer.optim.state_dict())
            trainer.save()

            try:
                # Close iterator
                sampler_test.close()
                sampler_train.close()
            except:
                pass

if __name__ == "__main__":
    '''
    Examples:
        Train a single model :
            python train.py --task train --bs 256 --epochs 1000 resnet34 dataset/outside
        Sweep hyper parameters :
            python train.py --task sweep --bs 256 --epochs 1000 resnet34 dataset/outside
        Train multiple models in sequence (must manually edit the models):
            python train.py --task queue --bs 256 --epochs 1000 resnet34 dataset/outside
    '''
    import argparse
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--task', type=str, help='Task to be done, could be [train, sweep, queue], default is train', default='train')
    parser.add_argument('--sweep_id', type=str, default=None, help='sweep id to load')
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
            if args.sweep_id:
                sweep_id = args.sweep_id
            else:
                sweep_id = wandb.sweep(
                  sweep=sweep_configuration, 
                  project='self-driving'
                  )
            wandb.agent(sweep_id, project ='self-driving', function=lambda: start_sweep(args.model, args.bs, args.epochs))
        case "queue":
            all_models = [
                "resnet34", 
                "resnet18",
                "cnn"
                #edit these models for your own use
            ]
            run = wandb.init(project='self-driving')
            start_queue(all_models, args.bs, args.epochs)
        case "train":
            run = wandb.init(project='self-driving')
            start_train(args.model, args.bs, args.epochs)
        case _:
            log.error("No action specified")
