import torch
import torchvision
import numpy as np
from gduap import l2_layer_loss, get_fooling_rate, get_data_loader, get_model

TF_PERTURBATIONS_DIR = '/cfarhomes/psando/Documents/UAPs/GD-UAP/classification/perturbations/'
PT_PERTURBATIONS_DIR = 'perturbations/'

def get_tf_uap(tf_uap_filename):
    """
    Given a .npy filename from the TensorFlow implementation, scale their
    imperceptibility contraint to ours (from [0,255] to [0,1]) and return the UAP tensor
    """
    tf_uap_fname = TF_PERTURBATIONS_DIR + tf_uap_filename
    tf_uap = np.load(tf_uap_fname).squeeze() / 255
    tf_uap_tensor = torch.as_tensor(np.transpose(tf_uap, (2, 0, 1))).unsqueeze(0)
    print(f"L-inf Norm: {torch.norm(tf_uap_tensor, p=np.inf)}")
    return tf_uap_tensor

def get_pt_uap(pt_uap_filename):
    """
    Given a .npy filename from the PyTorch implementation,
    return the UAP tensor
    """
    pt_uap_fname = PT_PERTURBATIONS_DIR + pt_uap_filename
    pt_uap = np.load(pt_uap_fname)
    pt_uap_tensor = torch.as_tensor(pt_uap)
    print(f"L-inf Norm: {torch.norm(pt_uap_tensor, p=np.inf)}")
    return pt_uap_tensor

def get_uap_loss(uap_filename, model_name, device, tf_impl):
    """
    Returns the loss of the TensorFlow UAP for comparison to ours
    """
    if tf_impl:
        uap = get_tf_uap(uap_filename)
    else:
        uap = get_pt_uap(uap_filename)
    uap = uap.to(device)
    model = get_model(model_name, device)
    loss = l2_layer_loss(model, uap)
    return loss.item()

def get_tf_uap_fooling_rate(uap_filename, model, device):
    """
    Obtain the fooling rate of the TensorFlow UAP on our final validation set
    which is ILSVRC 2012
    """
    uap = get_tf_uap(uap_filename)
    uap = uap.to(device)
    data_loader = get_data_loader('imagenet')
    tf_fooling_rate = get_fooling_rate(model, uap, data_loader, device, disable_tqdm=True)
    return tf_fooling_rate
