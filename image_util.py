import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def pic_to_square(picture):
    # todo resize to square
    pass


def load_pic_to_square_np(pic, size):
    picture = Image.open(pic).convert('L')  # opens and converts to grayscale
    if picture.width != picture.height:
        picture = pic_to_square(picture)
    picture = picture.resize((size, size), Image.ANTIALIAS)
    return np.array(picture) / 255


def show_image(image):
    image = np.clip(image, 0, 1)
    if image.shape[0] == 1:
        image.shape = image.shape[1:]
    if len(image.shape) < 3:
        image = np.expand_dims(image, 2)
        image = np.repeat(image, 3, 2)
    plt.imshow(image)
    plt.show(block=True)
