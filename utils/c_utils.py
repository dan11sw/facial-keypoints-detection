import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import constants.columns as cc
from tqdm import tqdm

def NaNMSELoss(output, target):
    filter_mark = ~torch.isnan(target)
    return ((output[filter_mark] - target[filter_mark]) ** 2).mean()

def get_image_and_keypoints(series):
    image = series[cc.COLUMN_IMAGE]
    image = np.fromstring(image, sep=' ').reshape([96, 96]) / 255.0
    keypoints = pd.DataFrame(series).drop(cc.COLUMN_IMAGE, axis=0).values.reshape([15, 2])

    return image, keypoints

def get_image_and_keypoints_by_id(train_data, id):
    series = train_data.iloc[id]
    return get_image_and_keypoints(series)

def show_image_with_keypoints(series):
    image, keypoints = get_image_and_keypoints(series)
    plt.imshow(image, cmap='gray')
    plt.plot(keypoints[:, 0], keypoints[:, 1], 'gx', color="red")
    plt.show()

def show_image_with_keypoints_by_id(train_data, id):
    series = train_data.iloc[id]
    show_image_with_keypoints(series)

def delete_duplicates(output_data=None, duplicates_list=None):
    if output_data is None:
        raise Exception("Error: argument output_data is None")
    elif duplicates_list is None:
        raise Exception("Error: argument duplicates_list is None")
    elif not isinstance(output_data, pd.DataFrame):
        raise Exception("Error: argument output_data is not instance of pd.DataFrame")

    for duplicate_df in tqdm(duplicates_list, desc="Processing duplicates"):
        n_rows = len(duplicate_df)
        keypoints_list = []
        image = None
        indexes = []

        for index, row in duplicate_df.iterrows():
            image = row[cc.COLUMN_IMAGE]
            indexes.append(index)

            keypoints_df = pd.DataFrame(row).drop([cc.COLUMN_IMAGE], axis=0).values
            keypoints_list.append(keypoints_df)

        keypoints_list = np.array(keypoints_list)
        keypoints_list = keypoints_list.reshape((n_rows, 30,))

        new_keypoints = np.nanmean(keypoints_list, dtype=np.float64, axis=0)
        index_lowest_lip = np.argmax(keypoints_list, axis=0)[-1]

        new_keypoints[-1] = keypoints_list[index_lowest_lip, -1]
        new_keypoints[-2] = keypoints_list[index_lowest_lip, -2]

        new_row = new_keypoints.tolist()
        new_row.append(image)

        output_data.drop(index=indexes, inplace=True)
        output_data.loc[max(output_data.index) + 1] = new_row

    print("\n✅ Duplicates removal completed!")
