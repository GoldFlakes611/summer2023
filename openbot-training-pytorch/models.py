'''
Name: models.py
Description: This program is used to define the neural networks using pytorch
            It also contains the ModelHub class which is used to load and save
            models. Helper functions such as get_model and get_model_openbot
            are for ease of loading models in seperate scripts
Date: 2023-08-25
Date Modified: 2023-08-25
'''
import torch
from torch import nn

class EdgeKernel(nn.Module):
    '''
    EdgeKernel class - used to define a convolutional layer that filters edges from images

    Args: 
        channels (int): number of channels in input image
        kernel (int): size of kernel to use for convolution

    Methods:
        forward(X): forward pass of the model
    '''
    def __init__(self, channels : int = 3, kernel : int = 3) -> None:
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

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model
        Args:
            X (torch.Tensor): input image
        Returns:
            X (torch.Tensor): output image
        '''
        X = torch.sqrt(self.Gx(X)**2 + self.Gy(X)**2)
        X = X / X.amax(dim=(2, 3), keepdim=True)
        return X


class CNN(nn.Module):
    ''' 
    CNN class - used to define a simple convolutional neural network

    Args:
        in_channels (int): number of channels in input image
        edge_filter (bool): whether to filter edges from input image
    
    Methods:
        forward(X): forward pass of the model
    '''
    NAME = "cnn"

    def __init__(self, in_channels : int = 3, edge_filter : bool = True) -> None:
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

    def forward(self, image : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of model
        Args:
            image (torch.Tensor): input image
        Returns:
            output (torch.Tensor): output of model
        '''
        image = self.filter(image)
        output = self.image_module(image)
        return output


class ModelHubBase(nn.Module):
    '''
    ModelHubBase Class - generic class for loading and using hub models, not to be used directly
                        but instead with get_model function

    Args:
        edge_filter (bool): whether to filter edges from input image
        old_model (bool): whether to use old model (for backward compatibility)

    Methods:
        forward(X): forward pass of the model
        load_model(model_name): load model from hub by name
    '''
    NAME = None
    def __init__(self, edge_filter : bool = True, old_model : bool = False) -> None:
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

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model
        Args:
            X (torch.Tensor): input image
        Returns:
            X (torch.Tensor): output image
        '''
        x = self.filter(x)
        x = self.model(x)
        #commented for conversion
        #if self.NAME == "googlenet":  # Fix ouput of googlenet
            #x = x[0]

        x = self.output(x)
        return x

    @classmethod
    def load_model(cls, model_name : str) -> nn.Module:
        '''
        Load model from hub by name
        Args:
            model_name (str): name of model
        Returns:
            model (ModelHubBase): model loaded from hub
        '''
        # This is a hack for compatibility
        # This method create a new subclass of ModelHubBase
        # and set the NAME attribute to the model_name
        cls_instance = type(model_name, (cls,), {})
        cls_instance.NAME = model_name
        return cls_instance


models = {
    "cnn": CNN,
}


def get_model(model_name):
    '''
    Get model by name from either the models dict or from hub
    Args:
        model_name (str): name of model
    Returns:
        model (nn.Module): model loaded from hub
    '''
    if model_name in models:
        return models[model_name]
    else:
        return ModelHubBase.load_model(model_name)
    
    
class AvgPool11(nn.Module):
    '''
    AvgPool11 Class - used to define a simple average pooling layer
    Args:
        None
    Methods:
        forward(X): forward pass of the model
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model
        Args:
            X (torch.Tensor): input image
        Returns:
            X (torch.Tensor): output image
        '''
        return x.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)


def get_model_openbot(model_name : str) -> nn.Module:
    '''
    Get model by name from either the models dict or from hub
    Then wrap it in a class that has outputs compatible with openbot
    Args:
        model_name (str): name of model
    Returns:
        model (nn.Module): model loaded from hub
    '''
    model_cls = get_model(model_name)
    class Model(model_cls):
        def forward(self, x):
            # Hotfix for openbot input and output format
            x = x.permute(0, 3, 1, 2)
            x = super().forward(x)
            x = x[..., [1, 0]]
            return x

    return Model
