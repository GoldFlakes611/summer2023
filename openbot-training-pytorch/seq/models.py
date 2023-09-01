from copy import deepcopy

import torch
from models import get_model as get_model_single
from torch import nn


class ViViTV2(nn.Module):
    NAME = "vivit"
    pretrained_vit = None

    def __init__(self, seq_len=10) -> None:
        super().__init__()
        vit = get_model_single("vit")()
        if not self.pretrained_vit is None:
            vit.load_state_dict(torch.load(self.pretrained_vit)["state"])

        vit = vit.model

        self.conv_proj = deepcopy(vit.conv_proj)
        self.encoder = deepcopy(vit.encoder)
        self.class_token = nn.Parameter(torch.clone(vit.class_token.data))

        pos_embedding = self.encoder.pos_embedding.data[:, 1:, :].repeat(1, seq_len, 1)
        pos_embedding = torch.cat([self.encoder.pos_embedding.data[:, 0:1, :], pos_embedding], dim=1)
        self.encoder.pos_embedding = nn.Parameter(pos_embedding)

        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2)
        )

    def forward(self, x):
        b, s, c, h, w = x.shape

        x = self.conv_proj(x.view(-1, c, h, w))
        _, f, n_h, n_w = x.shape
        x = x.view(b, s, f, n_h, n_w).permute(0, 1, 3, 4, 2).reshape(b, s * n_h * n_w, f)

        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.encoder.pos_embedding
        x = self.encoder(x)

        x = x[:, 0]
        x = self.head(x)

        return x
    

models = {
    "vivit": ViViTV2,
}


def get_model(name, config={}):
    cls = models[name]

    cls_instance = type(name, (cls,), {})

    if name in config:
        for k, v in  config.items():
            setattr(cls_instance, k, v)

    return cls_instance
