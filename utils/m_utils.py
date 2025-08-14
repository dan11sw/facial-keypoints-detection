import math
import cv2
import numpy as np

# Rotate point on define angle
def rotate_point(origin, point, angle):
    xo, yo = origin
    xp, yp = point

    x_final = xo + math.cos(math.radians(angle)) * (xp - xo) - math.sin(math.radians(angle)) * (yp - yo)
    y_final = yo + math.sin(math.radians(angle)) * (xp - xo) + math.cos(math.radians(angle)) * (yp - yo)

    return x_final, y_final


# Rotate image on define angle
def rotate_image(image, angle):
    # (96 / 2, 96 / 2)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # Get rotation matrix
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Use rotation matrix to kame rotate image
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)

    return result

# Translate image on define translate
def translate_image(image, translate):
    y, x = image.shape[0], image.shape[1]

    xp = x * translate[0]
    yp = y * translate[1]

    # Create matrix translation
    M = np.float32([[1, 0, xp], [0, 1, yp]])

    # Apply translation for this image
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted
