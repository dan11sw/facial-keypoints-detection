import torch
import matplotlib.pyplot as plt
import utils.c_utils as c_utils
import constants.columns as cc

def cuda_device_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPU: {num_gpus}")

        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB\n")

        current_device = torch.cuda.current_device()
        print(f"Current GPU: {current_device} ({torch.cuda.get_device_name(current_device)})")
    else:
        print("CUDA not available")

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    n_rows = len(transforms_list)
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    for idx, transformed_item in enumerate(transforms_list):
        transformed_image = transformed_item[cc.COLUMN_image]
        transformed_keypoints = transformed_item[cc.COLUMN_keypoint]

        # Print left image
        ax_left = axes[idx, 0]
        ax_left.imshow(image, cmap='gray')
        ax_left.plot(keypoints.reshape((15, 2))[:, 0],
                     keypoints.reshape((15, 2))[:, 1], 'gx')
        ax_left.axis('off')

        # Print right image
        ax_right = axes[idx, 1]
        ax_right.imshow(transformed_image, cmap='gray')
        ax_right.plot(transformed_keypoints.reshape((15, 2))[:, 0],
                      transformed_keypoints.reshape((15, 2))[:, 1], 'gx')
        ax_right.axis('off')

    plt.tight_layout()
    plt.show()
