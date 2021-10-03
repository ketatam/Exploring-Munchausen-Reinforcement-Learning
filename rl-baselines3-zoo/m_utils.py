import torch
from torch.nn import functional as F


def stable_scaled_log_softmax(x, tau, dim=-1):
    """Scaled log_softmax operation.
    Args:
    x: tensor of floats, inputs of the softmax (logits).
    tau: float, softmax temperature.
    dim: int, axis to perform the softmax operation.
    Returns:
    tau * log_softmax(x/tau, dim=dim)
    """
    max_x, _ = x.max(dim=dim, keepdim=True)
    y = x - max_x
    tau_lse = max_x + tau * torch.log(
        torch.sum(torch.exp(y / tau), dim=dim, keepdim=True)
    )
    return x - tau_lse


def stable_softmax(x, tau, dim=-1):
    """Stable softmax operation.
    Args:
    x: tensor of floats, inputs of the softmax (logits).
    tau: float, softmax temperature.
    dim: int, axis to perform the softmax operation.
    Returns:
    softmax(x/tau, dim=dim)
    """
    max_x, _ = x.max(dim=dim, keepdim=True)
    y = x - max_x
    return F.softmax(y / tau, dim=dim)
