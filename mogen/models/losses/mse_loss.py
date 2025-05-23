import torch.nn as nn
import torch.nn.functional as F
import torch
from kornia.filters.kernels import laplacian_1d

from ..builder import LOSSES
from .utils import weighted_loss


def gmof(x, sigma):
    """Geman-McClure error function."""
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


@weighted_loss
def mse_loss(pred, target):
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss_with_gmof(pred, target, sigma):
    """Extended MSE Loss with GMOF."""
    loss = F.mse_loss(pred, target, reduction="none")
    loss = gmof(loss, sigma)
    return loss


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.
    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        assert reduction in (None, "none", "mean", "sum")
        reduction = "none" if reduction is None else reduction
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        """Forward function of loss.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss


@LOSSES.register_module()
class LaplacianMSELoss(nn.Module):
    """
    Apply MSE loss to the 1D Laplacian of the input and target.
    """

    def __init__(self, reduction="mean", loss_weight=1.0, laplacian_kernel_size=3):
        super().__init__()
        self.mse_loss = MSELoss(reduction=reduction, loss_weight=loss_weight)
        self.reduction = reduction
        self.loss_weight = loss_weight

        assert laplacian_kernel_size > 0, "laplacian_kernel_size must be greater than 0"
        self.laplacian_kernel_size = laplacian_kernel_size
        # breakpoint()
        self.laplace_kernel = laplacian_1d(self.laplacian_kernel_size)[None, None, :]
        self.laplace_kernel.requires_grad = False

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        mask=None,
    ):
        """
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        # breakpoint()
        bs, t, f = pred.shape
        self.laplace_kernel = self.laplace_kernel.to(pred.device)
        pred_vec = pred.permute(0, 2, 1).reshape(-1, 1, t)  # (bs, f, t) -> (bs*f, 1, t)
        target_vec = target.permute(0, 2, 1).reshape(
            -1, 1, t
        )  # (bs, f, t) -> (bs*f, 1, t)
        pred_lap = F.conv1d(
            pred_vec, self.laplace_kernel, padding=self.laplacian_kernel_size // 2
        )
        target_lap = F.conv1d(
            target_vec, self.laplace_kernel, padding=self.laplacian_kernel_size // 2
        )

        pred_lap = pred_lap.reshape(bs, f, t).permute(
            0, 2, 1
        )  # (bs, f, t) -> (bs, t, f)
        target_lap = target_lap.reshape(bs, f, t).permute(
            0, 2, 1
        )  # (bs, f, t) -> (bs, t, f)

        if mask is not None:
            # breakpoint()
            # remove the boundary of mask to avoid edge effect
            kernel = torch.ones(1, 1, self.laplacian_kernel_size).to(mask.device)
            mask = F.conv1d(
                mask.unsqueeze(1).float(),
                kernel,
                padding=self.laplacian_kernel_size // 2,
            ).squeeze(1)
            lap_mask = (mask == self.laplacian_kernel_size).to(int)
        else:
            lap_mask = None
            # breakpoint()

        loss = self.mse_loss(
            pred_lap, target_lap, weight, avg_factor, reduction_override
        )
        return loss, lap_mask
