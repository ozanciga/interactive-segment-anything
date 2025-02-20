import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes


def smooth_binary_mask(mask, dilate=1, fill_holes=False, erode=1, 
                       kernel_size=3, smooth=True, closing=True):
    """
    Process a binary mask with optional dilation, hole filling, erosion, and smoothing.

    Args:
        mask (numpy.ndarray): Binary mask (1 for foreground, 0 for background).
        dilate (int): Number of dilation iterations (default: 0, disabled).
        fill_holes (bool): Whether to fill holes inside the mask (default: False).
        erode (int): Number of erosion iterations (default: 0, disabled).
        kernel_size (int): Size of the kernel for morphological operations (default: 3).
        smooth (bool): Apply Gaussian smoothing and re-thresholding to remove jagged edges.
        closing (bool): Perform morphological closing to fill gaps.

    Returns:
        numpy.ndarray: Processed binary mask.
    """
    processed_mask = mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilation
    if dilate > 0:
        processed_mask = cv2.dilate(processed_mask.astype(np.uint8), kernel, iterations=dilate)

    # Closing (Dilation followed by Erosion to remove small gaps)
    if closing:
        processed_mask = cv2.morphologyEx(processed_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Fill holes
    if fill_holes:
        processed_mask = binary_fill_holes(processed_mask.astype(np.uint8))

    # Erosion
    if erode > 0:
        processed_mask = cv2.erode(processed_mask.astype(np.uint8), kernel, iterations=erode)

    # Gaussian smoothing and thresholding to refine edges
    if smooth:
        blurred = cv2.GaussianBlur(processed_mask.astype(np.float32), (5, 5), 0)
        _, processed_mask = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)

    # Dilation
    if dilate > 0:
        processed_mask = cv2.dilate(processed_mask.astype(np.uint8), kernel, iterations=dilate)

    return processed_mask.astype(np.uint8)
