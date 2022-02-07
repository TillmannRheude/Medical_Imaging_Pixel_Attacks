import numpy as np

import random 
import torch

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
        self.std = random.uniform(0, 1)
        self.mean = random.uniform(0, 3)
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomGaussianBlur(object):
    def __init__(self):
        self.kernel_size = int(random.uniform(0, 4))
        self.sigma = random.uniform(0, 1)
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.kernel_size + self.sigma
    
    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, sigma={1})'.format(self.kernel_size, self.sigma)