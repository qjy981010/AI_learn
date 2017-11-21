import scipy.misc
import numpy as np

MEAN_VALUE = np.array([0.48, 0.458, 0.407]).reshape((1, 3, 1, 1))


def load_image(path):
    image = scipy.misc.imread(path)
    # normalize to 0-1
    image = image / 255.0
    image = np.reshape(image, ((1, image.shape[-1]) + image.shape[:-1])) - MEAN_VALUE

    return image


def save_image(image, path):
    img = image+MEAN_VALUE
    img = img[0]*255
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(path, img)
