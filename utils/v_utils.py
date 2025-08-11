import matplotlib.pyplot as plt
import utils.c_utils as c_utils

def show_count_images(duplicates_list=None, begin=0, end=6):
    if duplicates_list is None:
        raise Exception("Error: argument dlist is None")
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