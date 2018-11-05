'''
Testing the SIFT Implementation
Ameya Daigavane
5th November, 2018
'''

from SIFT.sift import SIFT
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(precision=1)

    SIFT_sample = SIFT()
    SIFT_sample.open_file('mike.jpg')

    print(SIFT_sample.image_grayscale_array)
    SIFT_sample.display_array(SIFT_sample.apply_blur(SIFT_sample.image_grayscale_array, scale=1))
    SIFT_sample.display_array(SIFT_sample.apply_blur(SIFT_sample.image_grayscale_array, scale=2))
    SIFT_sample.display_array(SIFT_sample.apply_blur(SIFT_sample.image_grayscale_array, scale=4))
    SIFT_sample.display_array(SIFT_sample.apply_blur(SIFT_sample.image_grayscale_array, scale=8))

    deriv = SIFT_sample.apply_derivative(SIFT.apply_blur(SIFT_sample.image_grayscale_array, 2))
    SIFT_sample.display_array(deriv[0])
    SIFT_sample.display_array(deriv[1])
    SIFT_sample.display_array(np.sqrt(np.square(deriv[0]) + np.square(deriv[1])))

    dog_octave, extrema_points = SIFT_sample.detect()
    SIFT_sample.display_array(SIFT_sample.rescale_array(dog_octave[0]))
    SIFT_sample.display_array(SIFT_sample.rescale_array(dog_octave[1]))
    SIFT_sample.display_array(SIFT_sample.rescale_array(dog_octave[2]))

    print(dog_octave[0])
    print(dog_octave[1])
    print(extrema_points)
