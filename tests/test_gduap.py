import pytest
import numpy as np
import torchvision.models as models

from gduap import *
from analyze import *

def test_get_rate_of_saturation_1():
    delta = np.zeros((1, 3, 224, 224))
    sat = get_rate_of_saturation(delta, 1)
    assert sat == 0

def test_get_rate_of_saturation_2():
    delta = np.ones((1, 3, 224, 224))
    delta[0][0][0][0] = 5
    sat = get_rate_of_saturation(delta, 5)
    assert sat == (1 / (3 * 224 * 224))

def test_normalize_1():
    image_batch = torch.zeros((1, 3, 5, 5))
    image_batch = normalize(image_batch)
    expected = torch.zeros((1, 3, 5, 5))
    expected[0][0] = (expected[0][0] - 0.485) / 0.229
    expected[0][1] = (expected[0][1] - 0.456) / 0.224
    expected[0][2] = (expected[0][2] - 0.406) / 0.225
    assert torch.equal(image_batch, expected)

def test_l2_layer_loss_1():
    # vgg16_no_data.npy has loss -154.91 and fooling rate of 50.16% on ImageNet validation set
    tf_uap_filename = 'vgg16_no_data.npy'
    model_name = 'vgg16'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf_loss = get_uap_loss(tf_uap_filename, model_name, device, tf_impl=True)
    assert np.round(tf_loss, 2) == -154.91
