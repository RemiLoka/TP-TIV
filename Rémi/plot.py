import main

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def show_square(roi,frame):
    im = Image.open(frame)
    fig, ax = plt.subplots()
    ax.imshow(im)
    rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    plt.show()

def show_hist(hist):
    plt.plot(hist)
    plt.title('Histogram of the Tracked Area')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()