import torch
import numpy as np
import constants.columns as cc


class CustomToTensor(object):
    def __call__(self, img_with_keypoints):
        image, keypoints = img_with_keypoints[cc.COLUMN_image], img_with_keypoints[cc.COLUMN_keypoint]

        image = np.transpose(image, (2, 0, 1)).copy()

        image = torch.from_numpy(image).type(torch.FloatTensor)
        keypoints = torch.from_numpy(keypoints).type(torch.FloatTensor)

        return {
            cc.COLUMN_image: image,
            cc.COLUMN_keypoint: keypoints
        }
