from torchvision.utils import _log_api_usage_once
import torch
import numpy as np
import constants.columns as cc
import utils.m_utils as m_utils


class CustomRandomRotation(object):
    __slots__ = ['angle', 'p']

    def __init__(self, angle: int, p=0.5):
        super().__init__()
        _log_api_usage_once(self)

        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")

        self.p = p
        self.angle = angle

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            angle = np.random.randint(-self.angle, self.angle)

            image, keypoints = img_with_keypoints[cc.COLUMN_image], img_with_keypoints[cc.COLUMN_keypoint]

            # Rotate image
            new_image = m_utils.rotate_image(image, -angle)

            width, height = image.shape[0], image.shape[1]
            origin = (width / 2, height / 2)
            new_keypoints = np.copy(keypoints)

            for i, point in enumerate(keypoints.reshape(15, 2)):
                new_point = m_utils.rotate_point(origin, point, angle)
                new_keypoints[i * 2] = new_point[0]
                new_keypoints[i * 2 + 1] = new_point[1]

            return {
                cc.COLUMN_image: new_image,
                cc.COLUMN_keypoint: new_keypoints
            }

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (p = {self.p})"

