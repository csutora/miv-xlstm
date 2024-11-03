import os

import wandb
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

def make_wandb_plot(x_data, y_data, x_title, y_title, title):
    """
    Creates wandb plot
    """
    table = wandb.Table(data=[[x,y] for (x,y) in zip(x_data, y_data)], columns=[x_title, y_title])
    return wandb.plot.line(table, x_title, y_title, title)

class WandbLogger():
    """
    This class was created in case of future development
    Currently it is only called once during start of training to log config to correct project
    Usage: run "wandb login" in your terminal and provide the API key
    """
    def __init__(self, config, project, on_off="offline"):
        wandb.login(key=os.getenv("WANDB_API_KEY", "."))
        wandb.init(
            project=project,
            mode=on_off,
            config=config # logs run config to wandb
        )

class FocalLoss(nn.Module):
    """
    Basic implementation for focal loss to not hard-code alpha and gamma values
    """
    def __init__(self, alpha, gamma = 2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha,
        self.gamma = gamma,
        self.reduction = reduction

    def forward(self, input, target):
        kwargs = {"alpha": self.alpha, "gamma": self.gamma, "reduction": self.reduction}
        return sigmoid_focal_loss(input, target, **kwargs)
