import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def plot_uap(uap):
    """
    Given a UAP with shape (1, 3, 224, 224) within the imperceptibility
    constraint, plot a normalized image
    """
    mean = torch.mean(uap, [2,3]).squeeze()
    std = torch.std(uap, [2,3]).squeeze()
    normalized_uap = (uap - mean[None,:,None,None]) / std[None,:,None,None]
    plot_uap = torchvision.utils.make_grid(normalized_uap.detach().cpu())
    plt.imshow(np.transpose(plot_uap, (1,2,0)))
