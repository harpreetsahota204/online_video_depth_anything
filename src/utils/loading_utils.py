import cv2
cv2.setNumThreads(1)
import numpy as np


def load_video_as_numpy(path):
    """Load a video file as a float32 numpy array in [0, 1].

    Parameters
    ----------
    path : str
        Path to the video file.

    Returns
    -------
    np.ndarray
        Shape (T, H, W, C), dtype float32, values in [0, 1].
    """
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return np.stack(frames)
