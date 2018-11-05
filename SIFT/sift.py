"""
Scale-Invariant Feature Transform: A Python Implementation with PIL
Ameya Daigavane
3rd November, 2018
"""

import sys
sys.path.append("..")

from cv import CV
from PIL import Image
import numpy as np


# The SIFT Class
class SIFT(CV):
    def __init__(self):
        super().__init__()
        self.image_marked = None

    # get boxed features on top of image
    def get_marked(self):
        return self.image_marked

    # obtain SIFT descriptors - the actual SIFT algorithm
    def detect(self, start_scale=2, octave_size=5):
        # 1. Scale-space Extrema Detection
        # Difference of Gaussians Pyramid
        prev_gaussian = CV.apply_blur(self.image_grayscale_array, start_scale)
        scale_factor = (2 ** (1/octave_size))
        scale = start_scale * scale_factor
        dog_octave = []
        for gaussian_count in range(octave_size):
            next_gaussian = CV.apply_blur(self.image_grayscale_array, scale)
            dog_octave.append(next_gaussian - prev_gaussian)
            prev_gaussian = next_gaussian
            scale *= scale_factor

        # Find extrema in DoG pyramid, excluding first and last scales
        extrema_points = []
        for dog_index, dog_matrix in enumerate(dog_octave[1:-1]):
            for i in range(1, dog_matrix.shape[0] - 1):
                for j in range(1, dog_matrix.shape[1] - 1):
                    # 8 neighbours in the same image
                    neighbour_pixels = []
                    neighbour_pixels.extend([dog_matrix[i - 1][j - 1], dog_matrix[i - 1][j], dog_matrix[i - 1][j + 1],
                                            dog_matrix[i][j - 1], dog_matrix[i][j + 1],
                                            dog_matrix[i + 1][j - 1], dog_matrix[i + 1][j], dog_matrix[i + 1][j + 1]])
                    # 9 neighbours below
                    dog_matrix_below = dog_octave[dog_index - 1]
                    neighbour_pixels.extend([dog_matrix_below[i - 1][j - 1], dog_matrix_below[i - 1][j], dog_matrix_below[i - 1][j + 1],
                                             dog_matrix_below[i][j - 1], dog_matrix_below[i][j], dog_matrix_below[i][j + 1],
                                             dog_matrix_below[i + 1][j - 1], dog_matrix_below[i + 1][j], dog_matrix_below[i + 1][j + 1]])

                    # 9 neighbours above
                    dog_matrix_above = dog_octave[dog_index + 1]
                    neighbour_pixels.extend([dog_matrix_above[i - 1][j - 1], dog_matrix_above[i - 1][j], dog_matrix_above[i - 1][j + 1],
                                             dog_matrix_above[i][j - 1], dog_matrix_above[i][j], dog_matrix_above[i][j + 1],
                                             dog_matrix_above[i + 1][j - 1], dog_matrix_above[i + 1][j], dog_matrix_above[i + 1][j + 1]])

                    # check if dog_matrix[i][j] is an extrema!
                    if dog_matrix[i][j] >= max(neighbour_pixels) or dog_matrix[i][j] <= min(neighbour_pixels):
                        extrema_points.append((i, j, dog_index + 1))

        return dog_octave, extrema_points


