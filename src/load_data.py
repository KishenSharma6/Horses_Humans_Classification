import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image

def preview_images(image_count, path, figsize = (20,20)):
    """Publishes matplotlib visualization showing randomly selected images from folder set by path variable.

    Args:
        image_count (int): number of images you would like randomly chosen for visualization
        path (string): string path to folder containing images for visualization on local machine
        figsize (tuple): tuple contain plot dimensions for matplotlib plot
    """
    plt.figure(figsize=figsize)

    target_data= path

    for i in range(image_count):
        file= random.choice(os.listdir(target_data))
        image_path= os.path.join(target_data, file)
        img= mpimg.imread(image_path)
        ax= plt.subplot(1,image_count, i+1)
        ax.title.set_text(file)
        plt.axis('off')
        plt.imshow(img)

def load_images(path):
    """Returns an array of loaded images from specified path on local machine

    Args:
        path (string): Path to directory containing images you would like loaded into a numpy array.
    """
    
    assert isinstance(path, str), 'Argument of wrong type!'
    
    images= []
    
    for i in glob.iglob(path):
        images.append(np.asarray(Image.open(i)))
    images= np.array(images)
    
    return images