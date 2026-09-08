"""Small runtime guards for the declared numerical execution contract."""
import torch


def autocast_enabled(device_type):
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:  # PyTorch releases before the device-type argument.
        return torch.is_autocast_enabled() or (device_type == "cpu" and torch.is_autocast_cpu_enabled())
