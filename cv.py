"""
The CV Class Definition: Standard Computer Vision Tools and Methods
Ameya Daigavane
5th November, 2018
"""

from PIL import Image
import numpy as np


# A generic CV class for specific CV algorithms to inherit from
class CV:
    def __init__(self):
        self.image_original = None
        self.image_grayscale = None
        self.image_original_array = None
        self.image_grayscale_array = None

    @staticmethod
    # creates a Gaussian filter matrix of shape (size, size) where size is the closest odd integer greater than 6*scale.
    def gaussian_blur_matrix(scale=1):
        size = 2*int(3*scale) + 1
        temp1 = np.square(np.abs(np.ceil(np.mgrid[-size / 2: size / 2, -size / 2: size / 2][0])))
        temp2 = np.square(np.abs(np.ceil(np.mgrid[-size / 2: size / 2, -size / 2: size / 2][1])))
        blur_matrix = np.exp(-(temp1 + temp2)/(2*(scale**2)))
        blur_matrix /= np.sum(blur_matrix)
        return blur_matrix

    @staticmethod
    # creates a pair of Sobel operator matrices of shape (3, 3) each.
    def sobel_matrix():
        operator_matrix = -np.mgrid[-1:2, -1:2]
        operator_matrix[1][1] *= 2
        operator_matrix[0] = operator_matrix[1]
        operator_matrix[1] = np.transpose(operator_matrix[1])
        return operator_matrix

    @staticmethod
    # 2D convolution of m2 with m1
    def convolve(m1, m2):
        padsize = max(m1.shape[0], m1.shape[1])
        m2_padded = CV.pad_value(m2, 128, padsize)
        m2_convolved = np.zeros(shape=m2.shape)

        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                xstart = i + m1.shape[0]//2
                ystart = j + m1.shape[1]//2
                m2_convolved[i][j] = np.sum(np.multiply(m1, m2_padded[xstart: xstart + m1.shape[0], ystart: ystart + m1.shape[1]]))

        return m2_convolved

    @staticmethod
    # pad array with value
    def pad_value(a, value=0, size=1):
        def pad_with(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = value
            vector[-pad_width[1]:] = value
            return vector

        return np.pad(a, size, pad_with)

    @staticmethod
    # displays an array representing grayscale entries as a grayscale image
    def display_array(a):
        grayscale_img = Image.fromarray(a)
        grayscale_img.show()

    @staticmethod
    # applies the gaussian blur of specified scale to the 2D image array
    def apply_blur(imgarray, scale=1):
        return CV.convolve(CV.gaussian_blur_matrix(scale), imgarray)

    @staticmethod
    # applies a Sobel derivative mask to the 2D image array
    def apply_derivative(imgarray):
        return np.array([CV.convolve(CV.sobel_matrix()[0], imgarray), CV.convolve(CV.sobel_matrix()[1], imgarray)])

    @staticmethod
    # rescales an array values in the range [0, 255], to obtain a grayscale representation of the image.
    # the array minimum gets mapped to 0 (black), and the maximum gets mapped to 255 (white).
    def rescale_array(a):
        return 255 * (a - np.min(a)) / (np.max(a) - np.min(a))

    # open a image file, associate PIL objects, and get array representations
    def open_file(self, image_path):
        self.image_original = Image.open(image_path)
        self.image_grayscale = self.image_original.convert('L')
        self.image_original.load()
        self.image_grayscale.load()
        self.image_original_array = np.asarray(self.image_original, dtype="uint32")
        self.image_grayscale_array = np.asarray(self.image_grayscale, dtype="uint32")

    # get the PIL Image object tied to the original image
    def get_original(self):
        return self.image_original

    # show the original image
    def show_original(self):
        self.image_original.show()

    # show the grayscale image
    def show_grayscale(self):
        self.image_grayscale.show()
