import numpy as np

import random 
import torch
import cv2


def select_random_pixels(height, width, seed=None):
    indeces = np.zeros((height * width, 2), dtype=np.uint)
    index = 0
    for y in range(height):
        for x in range(width):
            indeces[index] = [y,x]
            index +=1

    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indeces)

    return indeces

class AddRandomGaussianNoise(object):
    """ 
        Original source: 
        https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745 
        We slightly modified this one.
    """
    def __init__(self):
        self.std = random.uniform(0, 0.05)
        self.mean = random.uniform(0, 0.01)
        
    def __call__(self, img):
        if img.shape[0] == 3:
            img = np.rollaxis(img.detach().numpy(), 0, 3)
        else:
            img = img.detach().numpy()

        gauss = np.random.normal(self.mean, self.std, img.shape)
        noisy = img + gauss

        if img.shape[2] == 3:
            noisy = np.rollaxis(noisy, 2, 0)

        return torch.tensor(noisy)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomGaussianBlur(object):
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size
        
    def __call__(self, img):
        if img.shape[0] == 3:
            img = np.rollaxis(img.detach().numpy(), 0, 3)
        else:
            img = img.detach().numpy()

        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)

        if img.shape[2] == 3:
            img = np.rollaxis(img, 2, 0)

        return torch.tensor(img)

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, sigma={1})'.format(self.kernel_size, self.sigma)