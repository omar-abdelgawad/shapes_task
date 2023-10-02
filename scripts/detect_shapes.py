"""This module contains functions for detecting shapes in an image."""
from ultralytics import YOLO
from typing import Optional
from typing import Sequence
import numpy as np
import os

dir_path = os.path.dirname(__file__)
model_path = os.path.join(dir_path, "shapes_model", "last.pt")
model = YOLO(model_path)
all_scores = []
all_scores_max_length = 400


def calculate_score(results):
    shape_score_dict = {"circle": 20, "square": 15, "triangle": 10, "cross": 5}
    total_score = 0
    shapes_conversion_dict = results.names
    for obj in map(int, results.cpu().boxes.cls.int()):
        shape = shapes_conversion_dict[obj]
        total_score += shape_score_dict[shape]
    return total_score


def detect_shapes(img: np.ndarray) -> int:
    """Detects shapes in an image.

    Args:
        image: A numpy array representing an image.

    Returns:
        score(int):  representing the score of the shape detected.
    """

    raise NotImplementedError


def main(argv: Optional[Sequence[str]] = None) -> int:
    return 0


if __name__ == "__main__":
    exit(main(None))
