#==============================================================
# Name: models.py
# Desc: This program is used to define the convolution
# neural network using pytorch, it provides methods to use the
# model for training and inference
#==============================================================
import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class EdgeKernel(nn.Module):
    def __init__(self, channels=3, kernel=3) -> None:
        super().__init__()
        self.Gx = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=1)
        self.Gx.weight.data = torch.zeros(*self.Gx.weight.data.shape)
        self.Gy = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel, padding=1)
        self.Gy.weight.data = torch.zeros(*self.Gy.weight.data.shape)
        Kx = torch.as_tensor([[1,0,-1],[2,0,-2],[1,0,-1]])
        for i in range(channels):
            self.Gx.weight.data[i, i] = Kx
            self.Gy.weight.data[i, i] = Kx.T

        self.Gy.weight.requires_grad = False
        self.Gx.weight.requires_grad = False

    def forward(self, X):
        X = torch.sqrt(self.Gx(X)**2 + self.Gy(X)**2)
        X = X / X.amax(dim=(2, 3), keepdim=True)
        return X


class CNN(nn.Module):
    """ The Model class defines both the pytorch model
     and methods to train and it and run inference

    Attributes:
    -----------
    module : nn.Sequential
        Pytorch implementation of the model contains:
            -5 convolution layers
            -3 fully connected layers

        """
    NAME = "cnn"

    def __init__(self, in_channels=3, edge_filter=True):
        """Simple init function for class, defines model as series of sequential modules

        Parameters:
        -----------
        in_channels : int
            Number of channels of input tensor
        """
        super().__init__()

        self.filter = EdgeKernel(in_channels) if edge_filter else nn.Identity()

        self.image_module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.LazyConv2d(out_channels=96, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.LazyConv2d(out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.LazyConv2d(out_channels=256, kernel_size=3, stride=2, padding=0),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.LazyLinear(64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.LazyLinear(16),
            nn.ELU(),
            nn.LazyLinear(2),
        )

    def forward(self, image):
        """Forward pass of model

        Parameters:
        -----------
        image : torch.tensor
            image or batch of images in tensor form, after data augmentation and
            edge filtering

        Returns:
        --------
        torch.tensor -> output of neural net for each image in batch"""
        image = self.filter(image)
        output = self.image_module(image)
        return output


class ModelHubBase(nn.Module):
    NAME = None
    def __init__(self, edge_filter=True, old_model=False):
        super().__init__()
        self.filter = EdgeKernel(3) if edge_filter else nn.Identity()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', self.NAME)
        if old_model:
            # This is for backward compatibility with already trained models
            # Remove in the future
            self.output = nn.LazyLinear(2)
        else:
            if hasattr(self.model, "features"):
                # This fix some of the model in the following structure:
                # features -> adaptive_pooling -> classifier
                # the adaptive_pooling is not supported by onnx, so we need
                # define a new output layer
                self.model = self.model.features
                self.output = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    nn.Dropout(p=0.2),
                    nn.LazyLinear(128),
                    nn.ReLU(),
                    nn.Linear(128, 2),
                )
            else:
                self.output = nn.LazyLinear(2)

    def forward(self, x):
        x = self.filter(x)
        x = self.model(x)
        if self.NAME == "googlenet":  # Fix ouput of googlenet
            x = x[0]

        x = self.output(x)
        return x

    @classmethod
    def load_model(cls, model_name):
        # This is a hack for compatibility
        # This method create a new subclass of ModelHubBase
        # and set the NAME attribute to the model_name
        cls_instance = type(model_name, (cls,), {})
        cls_instance.NAME = model_name
        return cls_instance


class ViT(nn.Module):
    NAME = "vit"

    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.model.heads = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2)
        )

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.model.heads(x)

        return x


models = {
    "cnn": CNN,
    "vit": ViT,
}


def get_model(model_name):
    if model_name in models:
        return models[model_name]
    else:
        return ModelHubBase.load_model(model_name)
    
    
class AvgPool11(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)


def get_model_openbot(model_name):
    model_cls = get_model(model_name)
    class Model(model_cls):
        def forward(self, x):
            # Hotfix for openbot input and output format
            x = x.permute(0, 3, 1, 2)
            x = super().forward(x)
            x = x[..., [1, 0]]
            return x

    return Model
