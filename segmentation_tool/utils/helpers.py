import numpy as np
from PIL import Image
import cv2
import os
import shutil

def blend_mask_with_image(image, mask, color):
    """Blend the mask with the original image using a transparent color overlay."""
    mask_rgb = np.stack([mask * color[i] for i in range(3)], axis=-1)
    blended = (0.7 * image + 0.3 * mask_rgb).astype(np.uint8)
    return blended

def save_mask_as_png(mask, path):
    """Save the binary mask as a PNG."""
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(path)

def convert_mask_to_yolo(mask_path, image_path, class_id, output_path, append=False):
    """
    Convert a binary mask to YOLO-compatible segmentation labels.

    Args:
        mask_path (str): Path to the binary mask image.
        image_path (str): Path to the corresponding image.
        class_id (int): Class ID (e.g., 0 for void, 1 for chip).
        output_path (str): Path to save the YOLO label (.txt) file.
        append (bool): Whether to append labels to the file.

    Returns:
        None
    """
    try:
        # Load the binary mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found or invalid: {mask_path}")

        # Load the corresponding image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or invalid: {image_path}")

        h, w = image.shape[:2]  # Image height and width

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Determine file mode: "w" for overwrite or "a" for append
        file_mode = "a" if append else "w"

        # Open the output .txt file
        with open(output_path, file_mode) as label_file:
            for contour in contours:
                # Simplify the contour points to reduce the number of vertices
                epsilon = 0.01 * cv2.arcLength(contour, True)  # Tolerance for approximation
                contour = cv2.approxPolyDP(contour, epsilon, True)

                # Normalize contour points (polygon vertices)
                normalized_vertices = []
                for point in contour:
                    x, y = point[0]  # Extract x, y from the point
                    x_normalized = x / w
                    y_normalized = y / h
                    normalized_vertices.extend([x_normalized, y_normalized])

                # Write the polygon annotation to the label file
                if len(normalized_vertices) >= 6:  # At least 3 points required for a polygon
                    label_file.write(f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized_vertices) + "\n")

        print(f"YOLO segmentation label saved: {output_path}")

    except Exception as e:
        print(f"Error converting mask to YOLO format: {e}")
        raise RuntimeError(f"Failed to convert {mask_path} for class {class_id}: {e}")
