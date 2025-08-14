import matplotlib.pyplot as plt
import utils.c_utils as c_utils
import constants.columns as cc

def show_count_images(duplicates_list=None, begin=0, end=6):
    if duplicates_list is None:
        raise Exception("Error: argument duplicates_list is None")
    elif begin > end:
        raise Exception(f"Error: begin({begin}) more than end({end})")

    # Create one figure for all images
    n_rows = len(duplicates_list[0])
    n_cols = min(len(duplicates_list), (end - begin))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for col_idx, duplicate_df in enumerate(duplicates_list[begin:end]):
        for row_idx, (index, row) in enumerate(duplicate_df.iterrows()):
            image, keypoints = c_utils.get_image_and_keypoints(row)

            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap='gray')

            if row_idx == 1:
                ax.plot(keypoints[:, 0], keypoints[:, 1], 'gx', color="red")

            ax.axis('off')

    plt.tight_layout()
    plt.show()

def show_augment_images(image=None, keypoints=None, transforms_list=None):
    if image is None:
        raise Exception("Error: argument image is None")
    elif keypoints is None:
        raise Exception("Error: argument keypoints is None")
    elif transforms_list is None:
        raise Exception("Error: argument transforms_list is None")

    for transformed_image_with_keypoints in transforms_list:
        transformed_image = transformed_image_with_keypoints[cc.COLUMN_image]
        transformed_keypoints = transformed_image_with_keypoints[cc.COLUMN_keypoint]

        fig = plt.figure(figsize=(10, 20))
        fig.add_subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')

        # for i, keypoint_name in enumerate(columns_keypoints[::2]):
        # plt.annotate(keypoint_name[:-2],
        # (keypoints[2 * i], keypoints[2 * i + 1]),
        # fontsize='xx-small')

        plt.plot(keypoints.reshape((15, 2))[:, 0],
                 keypoints.reshape((15, 2))[:, 1], 'gx')

        fig.add_subplot(1, 2, 2)
        plt.imshow(transformed_image, cmap='gray')

        # for i, keypoint_name in enumerate(columns_keypoints[::2]):
        # plt.annotate(keypoint_name[:-2],
        # (transformed_keypoints[2 * i], transformed_keypoints[2 * i + 1]),
        # fontsize='xx-small')

        plt.plot(transformed_keypoints.reshape((15, 2))[:, 0],
                 transformed_keypoints.reshape((15, 2))[:, 1], 'gx')

        plt.show()

