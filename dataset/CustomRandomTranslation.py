from torchvision.utils import _log_api_usage_once
import torch
import numpy as np
import cv2
import constants.columns as cc

class CustomRandomTranslation(object):
    __slots__ = ['translate', 'p']

    def __init__(self, translate: (float, float), p=0.5):
        super().__init__()
        _log_api_usage_once(self)

        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")

        if not ((len(translate) == 2)
                and (0.0 <= translate[0] <= 1.0)
                and (0.0 <= translate[1] <= 1.0)):
            raise ValueError("there should be 2 numbers in translate, both between 0 and 1")

        self.p = p
        self.translate = translate

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            image, keypoints = img_with_keypoints[cc.COLUMN_image], img_with_keypoints[cc.COLUMN_keypoint]

            height, width = image.shape[0], image.shape[1]

            # Define border of translate
            x_translate_rate = np.random.uniform(low=-self.translate[0],
                                                 high=self.translate[0])
            y_translate_rate = np.random.uniform(low=-self.translate[1],
                                                 high=self.translate[1])

            x_translate_pixel = width * x_translate_rate
            y_translate_pixel = height * y_translate_rate

            M = np.float32([[1, 0, x_translate_pixel],
                            [0, 1, y_translate_pixel]])

            shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            new_keypoints = np.copy(keypoints)
            for i in range(len(keypoints) // 2):
                new_keypoints[2 * i] = keypoints[2 * i] + x_translate_pixel
                new_keypoints[2 * i + 1] = keypoints[2 * i + 1] + y_translate_pixel

            return {
                cc.COLUMN_image: shifted,
                cc.COLUMN_keypoint: new_keypoints
            }

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (p = {self.p})"

