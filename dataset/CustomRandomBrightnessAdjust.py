from torchvision.utils import _log_api_usage_once
import torch
import numpy as np
import cv2
import constants.columns as cc

class CustomRandomBrightnessAdjust(object):
    __slots__ = ['brightness', 'p']

    def __init__(self, brightness: float, p=0.5):
        super().__init__()
        _log_api_usage_once(self)

        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")

        if not (0.0 <= brightness <= 1.0):
            raise ValueError("brightness should be float between 0 and 1")

        self.p = p
        self.brightness = brightness

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            image, keypoints = img_with_keypoints[cc.COLUMN_image], img_with_keypoints[cc.COLUMN_keypoint]
            random_brightness = np.random.uniform(low=-self.brightness,
                                                  high=self.brightness)

            hsv = cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            lim = 255 - random_brightness
            v[v > lim] = 255
            v[v <= lim] += random_brightness

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            img = np.clip(img, 0.0, 1.0)

            return {
                cc.COLUMN_image: img,
                cc.COLUMN_keypoint: keypoints
            }

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (p = {self.p})"
