import numpy as np
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


