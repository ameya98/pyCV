"""
Scale-Invariant Feature Transform: A Python Implementation with PIL
Ameya Daigavane
3rd November, 2018
"""

import sys
sys.path.append("..")

from cv import CV
from PIL import Image, ImageDraw
import numpy as np


# The SIFT Class
class SIFT(CV):
    def __init__(self):
        super().__init__()
        self.image_marked = None

    # Mark points passed as coordinate tuples on the image.
    def mark_points(self, points, dims=(5, 5), fill="red"):
        self.image_marked = self.image_original.copy()
        draw = ImageDraw.Draw(self.image_marked)
        for point in points:
            draw.rectangle([(point[0] - dims[0]/2, point[1] - dims[1]/2),
                            (point[0] + dims[0]/2, point[1] + dims[1]/2)], fill=fill)

    # Get boxed features on top of image
    def get_marked(self):
        return self.image_marked

    # Show boxed features on top of image
    def show_marked(self):
        self.image_marked.show()

    # obtain SIFT descriptors - the actual SIFT algorithm
    def detect(self, start_scale=2, octave_size=5, edge_threshold=10, contrast_threshold=0.03):

        # Construct Difference-of-Gaussians (DoG) Pyramid
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
        for temp_index, dog_matrix in enumerate(dog_octave[1:-1]):
            for i in range(1, dog_matrix.shape[0] - 1):
                for j in range(1, dog_matrix.shape[1] - 1):

                    # Actual index in the DoG pyramid
                    dog_index = temp_index + 1

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

                    # Check if dog_matrix[i][j] is an extrema!
                    if dog_matrix[i][j] > max(neighbour_pixels) or dog_matrix[i][j] < min(neighbour_pixels):
                        extrema_points.append((j, i, dog_index))

        Find derivatives for all scales in octave
        dog_octave_deriv = [SIFT.apply_derivative(dog_image) for dog_image in dog_octave]
        dog_octave_deriv_x = [deriv[0] for deriv in dog_octave_deriv]
        dog_octave_deriv_y = [deriv[1] for deriv in dog_octave_deriv]
        temp_doublederiv = [SIFT.apply_derivative(dog_image) for dog_image in dog_octave_deriv_y]

        dog_octave_doublederiv_xy = [doublederiv[0] for doublederiv in temp_doublederiv]
        dog_octave_doublederiv_yy = [doublederiv[1] for doublederiv in temp_doublederiv]
        dog_octave_doublederiv_xx = [SIFT.apply_derivative(dog_image)[0] for dog_image in dog_octave_deriv_x]

        # Rescale all scales in octave to range 0 to 255.
        dog_octave_rescaled = [SIFT.rescale_array(dog_image) for dog_image in dog_octave]

        # Apply intensity test
        passed_extrema_points = []
        for extrema_point in extrema_points:
            i, j, index = extrema_point

            if dog_octave_rescaled[index][i][j] > contrast_threshold * 255:
                passed_extrema_points.append(extrema_point)

        extrema_points = passed_extrema_points

        # Apply gradient test - evaluate Hessian at extrema points
        passed_extrema_points = []
        for extrema_point in extrema_points:
            i, j, index = extrema_point

            hessian = np.asarray([[dog_octave_doublederiv_xx[index][i][j], dog_octave_doublederiv_xy[index][i][j]],
                                  [dog_octave_doublederiv_xy[index][i][j], dog_octave_doublederiv_yy[index][i][j]]])

            if (np.trace(hessian) ** 2) < np.linalg.det(hessian)*((edge_threshold + 1)**2)/edge_threshold :
                passed_extrema_points.append(extrema_point)

        extrema_points = passed_extrema_points

        # Mark keypoints as 10px-by-10px rectangles on image
        self.mark_points(extrema_points, dims=(5, 5))

        return dog_octave, extrema_points
