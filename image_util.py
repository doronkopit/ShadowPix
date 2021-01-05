import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2



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


def grad_conv(pics,kernel_size=3):
    if type(pics) != np.ndarray or len(pics.shape) > 2:
        g_x = np.array([cv2.Sobel(pic, cv2.CV_64F, 1, 0, ksize=kernel_size) for pic in pics])
        g_y = np.array([cv2.Sobel(pic, cv2.CV_64F, 0, 1, ksize=kernel_size) for pic in pics])
    else:
        g_x = cv2.Sobel(pics, cv2.CV_64F, 1, 0, ksize=kernel_size)
        g_y = cv2.Sobel(pics, cv2.CV_64F, 0, 1, ksize=kernel_size)

    return np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))


def lp_conv(pics,kernel_size=3):
    if type(pics) != np.ndarray or len(pics.shape) > 2:
        return np.array([cv2.blur(pic, (kernel_size, kernel_size)) for pic in pics])
    return cv2.blur(pics, (kernel_size, kernel_size))