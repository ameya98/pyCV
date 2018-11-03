"""
Scale-Invariant Feature Transform: A Python Implementation with PIL
Ameya Daigavane
3rd November, 2018
"""

from PIL import Image
import numpy as np


class SIFT:
    def __init__(self):
        self.image_original = None
        self.image_grayscale = None
        self.image_original_array = None
        self.image_grayscale_array = None
        self.image_marked = None

    @staticmethod
    # creates a gaussian blur matrix of shape (size, size) where size is the closest odd integer to 6*scale.
    def gaussian_blur_matrix(scale=1):
        size = 2*int(3*scale) + 1
        temp1 = np.square(np.abs(np.ceil(np.mgrid[-size / 2: size / 2, -size / 2: size / 2][0])))
        temp2 = np.square(np.abs(np.ceil(np.mgrid[-size / 2: size / 2, -size / 2: size / 2][1])))
        blur_matrix = np.exp(-(temp1 + temp2)/(2*(scale**2)))
        blur_matrix /= np.sum(blur_matrix)
        return blur_matrix

    @staticmethod
    # 2D convolution of m2 with m1
    def convolve(m1, m2):
        padsize = max(m1.shape[0], m1.shape[1])
        m2_padded = SIFT.pad_zeros(m2, padsize)
        m2_convolved = np.zeros(shape=m2.shape)

        print(m1.shape, m2.shape, m2_padded.shape)
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                xstart = i + m1.shape[0]//2
                ystart = j + m1.shape[1]//2
                m2_convolved[i][j] = np.sum(np.multiply(m1, m2_padded[xstart: xstart + m1.shape[0], ystart: ystart + m1.shape[1]]))

        return m2_convolved

    @staticmethod
    # Pad with zeros
    def pad_zeros(a, size=1):
        def pad_with(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector

        return np.pad(a, size, pad_with)

    @staticmethod
    # displays an array representing grayscale entries as a grayscale image
    def display_array(a):
        grayscale_img = Image.fromarray(a)
        grayscale_img.show()

    # open a image file, associate PIL objects, and get array representations
    def open_file(self, image_path):
        self.image_original = Image.open(image_path)
        self.image_grayscale = self.image_original.convert('L')
        self.image_original.load()
        self.image_grayscale.load()
        self.image_original_array = np.asarray(self.image_original, dtype="uint32")
        self.image_grayscale_array = np.asarray(self.image_grayscale, dtype="uint32")

    def detect(self):
        # Scale-space Extrema Detection - Difference of Gaussians
        m1 = self.convolve(SIFT.gaussian_blur_matrix(1), self.image_grayscale_array)
        m2 = self.convolve(SIFT.gaussian_blur_matrix(2), self.image_grayscale_array)

        return m1

    # applies the gaussian blur of specified scale to the grayscale image
    def apply_blur(self, scale=1):
        return self.convolve(SIFT.gaussian_blur_matrix(scale), self.image_grayscale_array)


    # get the PIL Image object tied to the original image
    def get_original(self):
        return self.image_original

    # show the original image
    def show_original(self):
        self.image_original.show()

    def show_grayscale(self):
        self.image_grayscale.show()

    # get boxed features on top of image
    def get_marked(self):
        return self.image_marked


if __name__ == "__main__":
    np.set_printoptions(precision=1)

    SIFT_sample = SIFT()
    SIFT_sample.open_file('mike.jpg')

    print(SIFT_sample.image_grayscale_array)

    SIFT_sample.show_grayscale()
    SIFT_sample.display_array(SIFT_sample.apply_blur(2))
