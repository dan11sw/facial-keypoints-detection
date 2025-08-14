from torchvision.utils import _log_api_usage_once
import torch
import numpy as np
import constants.columns as cc


class CustomRandomVerticalFlip(object):
    __slots__ = ['p']

    def __init__(self, p=0.5):
        super().__init__()
        _log_api_usage_once(self)

        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")

        self.p = p

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            image, keypoints = img_with_keypoints[cc.COLUMN_image], img_with_keypoints[cc.COLUMN_keypoint]

            # array reflection from left to right
            new_image = np.fliplr(image)

            # Keypoints
            width_img = image.shape[0]
            old_keypoints = keypoints
            new_keypoints = np.copy(keypoints)
            # left eye center x, y
            new_keypoints[0] = width_img - old_keypoints[2]
            new_keypoints[1] = old_keypoints[3]
            # right eye center x, y
            new_keypoints[2] = width_img - old_keypoints[0]
            new_keypoints[3] = old_keypoints[1]
            # left eye inner corner x, y
            new_keypoints[4] = width_img - old_keypoints[8]
            new_keypoints[5] = old_keypoints[9]
            # left eye outer corner x, y
            new_keypoints[6] = width_img - old_keypoints[10]
            new_keypoints[7] = old_keypoints[11]
            # right eye inner corner x, y
            new_keypoints[8] = width_img - old_keypoints[4]
            new_keypoints[9] = old_keypoints[5]
            # right eye outer corner x, y
            new_keypoints[10] = width_img - old_keypoints[6]
            new_keypoints[11] = old_keypoints[7]
            # left eyebrow inner end x, y
            new_keypoints[12] = width_img - old_keypoints[16]
            new_keypoints[13] = old_keypoints[17]
            # left eyebrow outer end x, y
            new_keypoints[14] = width_img - old_keypoints[18]
            new_keypoints[15] = old_keypoints[19]
            # right eyebrow inner end x, y
            new_keypoints[16] = width_img - old_keypoints[12]
            new_keypoints[17] = old_keypoints[13]
            # right eyebrow outer end x, y
            new_keypoints[18] = width_img - old_keypoints[14]
            new_keypoints[19] = old_keypoints[15]
            # nose tip x, y
            new_keypoints[20] = width_img - old_keypoints[20]
            # new_keypoints[21] = old_keypoints[21]
            # mouth left corner x, y
            new_keypoints[22] = width_img - old_keypoints[24]
            new_keypoints[23] = old_keypoints[25]
            # mouth right corner x, y
            new_keypoints[24] = width_img - old_keypoints[22]
            new_keypoints[25] = old_keypoints[23]
            # mouth center top lip x, y
            new_keypoints[26] = width_img - old_keypoints[26]
            # new_keypoints[27] = old_keypoints[27]
            # mouth center bottom lip x, y
            new_keypoints[28] = width_img - old_keypoints[28]
            # new_keypoints[29] = old_keypoints[29]

            return {cc.COLUMN_image: new_image, cc.COLUMN_keypoint: new_keypoints}

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p = {self.p})"
