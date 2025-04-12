import cv2
import numpy as np

def detect_motion(prev_frame, current_frame, threshold=25):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Frame differencing
    frame_diff = cv2.absdiff(prev_gray, current_gray)

    # Thresholding to binarize
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels (motion level)
    motion_score = np.sum(thresh)

    return motion_score > 50000  # Motion threshold (adjust as needed)
