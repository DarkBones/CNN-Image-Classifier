import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import numpy as np

# plots the image and its probabilities
def plot_image_probs(imagepath, h, categories, threshold=0):
    plt.rcdefaults()

    # plot the image
    img = mpimg.imread(imagepath)
    fig = plt.figure()
    a = fig.add_subplot(2,2,1)
    a = plt.imshow(img)
    plt.axis('off')

    # plot the probabilities
    y_pos = np.arange(len(categories))
    b = fig.add_subplot(2,2,2)
    b.barh(y_pos, h, align='center', color='grey')
    b.set_yticks(y_pos)
    b.set_yticklabels(categories)
    b.invert_yaxis()
    b.set_xlabel('Probability')
    if threshold > 0:
        b.axvline(threshold, ls='--', color='r')
    plt.xlim(0, 1.0)

    plt.show()
    return