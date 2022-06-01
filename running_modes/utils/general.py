import time

import numpy as np
import torch


def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def set_default_device_cuda(dont_use_cuda=False):
    """Sets the default device (cpu or cuda) used for all tensors."""
    if torch.cuda.is_available() == False or dont_use_cuda:
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return False
    else:  # device_name == "cuda":
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return True


def estimate_run_time(start_time, n_steps, step):
    time_elapsed = int(time.time() - start_time)
    time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
    summary = {"elapsed": time_elapsed, "left": time_left}
    return summary