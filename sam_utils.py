import matplotlib.pyplot as plt
import numpy as np
from torch import inference_mode
from morphology_utils import smooth_binary_mask


def show_mask(mask: np.ndarray, ax: plt.Axes, random_color: bool = True) -> None:
    """
    Displays a smoothed binary mask on a given matplotlib axis.

    Args:
        mask (np.ndarray): Binary mask to display.
        ax (plt.Axes): Matplotlib axis to plot the mask on.
        random_color (bool): Whether to use a random color or a predefined blue color.
    """
    # Generate color for the mask
    color = np.concatenate([np.random.random(3), [0.6]]) if random_color else np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    # Apply smoothing and morphological operations to the mask
    mask = smooth_binary_mask(
        mask,
        dilate=3,
        fill_holes=True,
        erode=6,
        kernel_size=1,
        smooth=True,
        closing=True
    )

    # Reshape and display the mask with transparency
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_masks_on_image(raw_image: np.ndarray, masks: np.ndarray, scores: np.ndarray) -> None:
    """
    Displays masks over the raw image with their respective confidence scores.

    Args:
        raw_image (np.ndarray): Original image.
        masks (np.ndarray): Predicted masks.
        scores (np.ndarray): Corresponding confidence scores.
    """
    scores = scores.squeeze()

    # Ensure scores are 2D for consistent processing
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)

    batch_size, num_predictions = scores.shape
    fig, axes = plt.subplots(batch_size, num_predictions, figsize=(num_predictions * 6, batch_size * 4))

    # Ensure axes is a 2D array to handle varying batch sizes and predictions
    axes = np.atleast_2d(axes)

    for b in range(batch_size):
        # Sort masks by their scores in descending order
        index_order = np.argsort(-scores[b].cpu().numpy())

        for i, index in enumerate(index_order):
            ax = axes[b, i] if batch_size > 1 or num_predictions > 1 else axes
            ax.imshow(np.array(raw_image))
            show_mask(masks[b, index].cpu().numpy(), ax)
            ax.set_title(f"Mask {i + 1}, Score: {100 * scores[b, index]:.1f}%")
            ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    plt.show()


def get_mask_and_scores_from_sam_input(
    sam_model, sam_processor, raw_image: np.ndarray, image_embeddings, 
    input_points=None, input_boxes=None
):
    """
    Retrieves masks and their scores from a SAM model given image embeddings and optional inputs.

    Args:
        sam_model: Pre-trained SAM model.
        sam_processor: SAM processor for input preprocessing.
        raw_image (np.ndarray): Input image.
        image_embeddings: Image embeddings for the SAM model.
        input_points (list, optional): Points to guide mask prediction.
        input_boxes (list, optional): Bounding boxes to guide mask prediction.

    Returns:
        tuple: Processed masks and their corresponding IOU scores.
    """
    if input_points:
        inputs = sam_processor(raw_image, input_points=[input_points], return_tensors="pt").to('cuda')
    elif input_boxes:
        inputs = sam_processor(raw_image, input_boxes=[input_boxes], return_tensors="pt").to('cuda')
    else:
        raise ValueError("Either input_points or input_boxes must be provided.")

    # Remove unnecessary pixel values and add image embeddings
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    # Perform inference without gradient calculation
    with inference_mode():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    return masks, outputs.iou_scores


def sample_points_from_box(boxes: list, num_samples_per_box: int = 4) -> list:
    """
    Samples points from inside the provided bounding boxes using a normal distribution centered at the box's center.

    Args:
        boxes (list): List of bounding boxes, each defined by [x1, y1, x2, y2].
        num_samples_per_box (int): Number of points to sample per box.

    Returns:
        list: List of sampled points for each box.
    """
    sampled_points = []

    for box in boxes:
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        std_x, std_y = (x2 - x1) / 6, (y2 - y1) / 6  # 99.7% of samples within the box bounds

        points = []
        while len(points) < num_samples_per_box:
            x_sample, y_sample = np.random.normal(center_x, std_x), np.random.normal(center_y, std_y)
            if x1 <= x_sample <= x2 and y1 <= y_sample <= y2:
                points.append((int(x_sample), int(y_sample)))

        sampled_points.append(points)

    return sampled_points
