
# Libraries Import
import cv2
import numpy as np
import os

# Helper Function
def split_extension(path, suffix=''):
    """
    Inserts a suffix between the filename and its extension.

    Args:
        path: Full path or filename (e.g., 'myfile.txt').
        suffix: String to insert before the extension (default: empty string).

    Returns:
        Modified path or filename with the suffix inserted before the extension.
    """
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"

# Box Utility Class
class Box:
    class BoxSource:
        Torch = 'torch'
        Numpy = 'numpy'

    @staticmethod
    def box2box(box, in_source, to_source, return_int=True):
        """
        Converts a bounding box from one format/source to another.
        Assumes input is already in x1, y1, x2, y2 format (e.g., YOLOv8 post-NMS).

        Args:
            box: Bounding box coordinates.
            in_source: Source format of the input box (e.g., 'torch', 'numpy').
            to_source: Target format for the output box.
            return_int: If True, returns coordinates as integers (default: True).

        Returns:
            Converted bounding box as a list of coordinates.
        """
        b = [float(x) for x in box]
        if return_int:
            b = [int(x) for x in b]
        return b

    @staticmethod
    def put_box(img, bbox):
        """
        Draws a bounding box on the given image.

        Args:
            img: Input image (numpy array).
            bbox: Bounding box coordinates as [x1, y1, x2, y2].

        Returns:
            Image with the bounding box drawn.
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img

    @staticmethod
    def put_text(img, text, pos):
        """
        Draws text on the given image at the specified position.

        Args:
            img: Input image (numpy array).
            text: Text to draw.
            pos: Position (x, y) where the text should be drawn.

        Returns:
            Image with the text drawn.
        """
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        return img

    @staticmethod
    def fill_outer_box(heatmap, bbox):
        """
        Masks everything outside the given bounding box in the heatmap.

        Args:
            heatmap: Input heatmap (numpy array).
            bbox: Bounding box coordinates as [x1, y1, x2, y2].

        Returns:
            Heatmap with values outside the box set to zero.
        """
        mask = np.zeros_like(heatmap)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = heatmap[y1:y2, x1:x2]
        return mask
