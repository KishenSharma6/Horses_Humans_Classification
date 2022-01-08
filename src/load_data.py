import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image

def preview_images(data, num_images, figsize):
    """Publishes matplotlib visualization showing randomly selected images from tensor

    Args:
        data (np.array): Contains array of image data
        num_images:number of images you would like randomly chosen for visualization
        figsize (tuple): tuple contain plot dimensions for matplotlib plot
    """
    
    assert num_images>0, "Number must be greater than 0"
    plt.figure(figsize=figsize)

    rand_index= np.random.randint(0, len(data), num_images)

    for i, value in enumerate(rand_index):
        ax= plt.subplot(1, num_images, i+1)
        plt.axis('off')
        plt.imshow(data[value])

def load_images(path):
    """Returns an array of loaded images from specified path on local machine

    Args:
        path (string): Path to directory containing images you would like loaded into a numpy array.
    """
    
    assert isinstance(path, str), 'Argument of wrong type!'
    
    images= []
    
    for i in glob.iglob(path + "/*.png"):
        images.append(np.asarray(Image.open(i)))
    images= np.array(images)
    
    return images