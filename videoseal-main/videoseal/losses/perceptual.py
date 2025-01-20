# videoseal/losses/perceptual.py

import torch
import torch.nn as nn
from lpips import LPIPS

from .watson_fft import ColorWrapper, WatsonDistanceFft
from .watson_vgg import WatsonDistanceVgg
from .dists import DISTS
from .jndloss import JNDLoss
from .focal import FocalFrequencyLoss
from .ssim import SSIM, MSSSIM
from .yuvloss import YUVLoss


def build_loss(loss_name):
    if loss_name == "none":
        return NoneLoss()
    elif loss_name == "lpips":
        return LPIPS(net="vgg").eval()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "yuv":
        return YUVLoss()
    elif loss_name == "focal":
        return FocalFrequencyLoss()
    elif loss_name == "ssim":
        return SSIM()
    elif loss_name == "msssim":
        return MSSSIM()
    elif loss_name == "jnd":
        return JNDLoss(loss_type=0)
    elif loss_name == "jnd2":
        return JNDLoss(loss_type=2)
    elif loss_name == "dists":
        # Ensure the path to DISTS checkpoint is correct
        return DISTS("/path/to/dists_ckpt.pth").eval()
    elif loss_name == "watson_vgg":
        # Ensure the path to Watson VGG checkpoint is correct
        model = WatsonDistanceVgg(reduction="none")
        ckpt_loss = "/path/to/rgb_watson_vgg_trial0.pth"
        model.load_state_dict(torch.load(ckpt_loss))
        return model
    elif loss_name == "watson_dft":
        # Ensure the path to Watson FFT checkpoint is correct
        model = ColorWrapper(WatsonDistanceFft, (), {"reduction": "none"})
        ckpt_loss = "/path/to/rgb_watson_fft_trial0.pth"
        model.load_state_dict(torch.load(ckpt_loss))
        return model
    else:
        raise ValueError(f"Loss type {loss_name} not supported.")


class NoneLoss(nn.Module):
    def forward(self, x, y):
        return torch.zeros(1, requires_grad=True)


class PerceptualLoss(nn.Module):
    def __init__(
        self, 
        percep_loss: str
    ):
        super(PerceptualLoss, self).__init__()

        self.percep_loss = percep_loss
        self.perceptual_loss = self.create_perceptual_loss(percep_loss)

    def create_perceptual_loss(
        self, 
        percep_loss: str
    ):
        """
        Create a perceptual loss function from a string.
        Args:
            percep_loss: (str) The perceptual loss string.
                Example: "lpips", "lpips+mse", "0.1_lpips+0.9_mse", ...
        """
        # Split the string into the different losses
        parts = percep_loss.split('+')

        # Initialize losses as an nn.ModuleDict
        self.losses = nn.ModuleDict()

        # Populate self.losses with all specified losses
        for part in parts:
            if '_' in part:  # Format: 'weight_loss'
                weight, loss_key = part.split('_')
                weight = float(weight)
            else:  # Default weight = 1
                weight, loss_key = 1.0, part

            # Build and add the loss to self.losses
            self.losses[loss_key] = build_loss(loss_key)
        
        # Define the combined loss function
        def combined_loss(x, y):
            total_loss = 0.0
            for part in parts:
                if '_' in part:
                    weight, loss_key = part.split('_')
                    weight = float(weight)
                else:
                    weight, loss_key = 1.0, part
                total_loss += weight * self.losses[loss_key](x, y).mean()
            return total_loss

        return combined_loss

    def forward(
        self, 
        imgs: torch.Tensor,
        imgs_w: torch.Tensor,
    ) -> torch.Tensor:
        return self.perceptual_loss(imgs, imgs_w)

    def to(self, device, *args, **kwargs):
        """
        Override the to method to move all perceptual loss functions to the device.
        """
        super().to(device)
        self.losses.to(device)
        return self

    def __repr__(self):
        return f"PerceptualLoss(percep_loss={self.percep_loss})"
