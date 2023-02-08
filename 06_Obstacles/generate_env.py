"""
Generate a shape_maze
"""
import numpy as np
from skimage.draw import random_shapes


# https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.random_shapes


def random_shape_maze(width,
                      height,
                      max_shapes,  # Max number of shapes
                      max_size,
                      allow_overlap,
                      shape=None):

    x, _ = random_shapes((height, width),
                         max_shapes,
                         max_size=max_size,
                         channel_axis=None,  # Gray scale
                         shape=shape,
                         allow_overlap=allow_overlap)

    x[x == 255] = 0
    x[np.nonzero(x)] = 1

    return x


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    grid_size = 30

    x = random_shape_maze(width=grid_size,
                          height=grid_size,
                          max_shapes=grid_size / 10,  # Max number of shapes
                          max_size=grid_size / 5,
                          allow_overlap=True,
                          shape=None)
    print(x, x.max(), x.min())

    plt.imshow(x)
    plt.show()
