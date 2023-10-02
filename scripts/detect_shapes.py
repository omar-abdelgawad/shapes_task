"""This module contains function(s) for detecting shapes in an image. The function to import is called detect_shapes."""
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


def _calculate_score(results):
    shape_score_dict = {"circle": 20, "square": 15, "triangle": 10, "cross": 5}
    total_score = 0
    shapes_conversion_dict = results.names
    for obj in map(int, results.cpu().boxes.cls.int()):
        shape = shapes_conversion_dict[obj]
        total_score += shape_score_dict[shape]
    return total_score


def detect_shapes(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Detects shapes in an image.

    Args:
        image(np.ndarray): input frame.

    Returns:
        labeled_img(np.ndarray): image with detected shapes.
        score(int): total score of all detected shapes.
    """
    global all_scores
    results = model(img, conf=0.5, show=False)[0]
    labeled_img = results.plot(labels=False)
    score = _calculate_score(results)
    all_scores.append(score)
    if len(all_scores) > all_scores_max_length:
        all_scores.pop(0)
    score = int(np.median(all_scores))
    return labeled_img, score


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main function for testing shape detection in a video stream."""
    import cv2

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img, score = detect_shapes(frame)
        cv2.putText(
            img,
            f"Score: {score}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("det", img)
        if cv2.waitKey(1) == ord("d"):
            break
    cv2.destroyAllWindows()
    cap.release()
    return 0


if __name__ == "__main__":
    exit(main(None))
