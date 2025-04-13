import cv2
import numpy as np
from collections import deque

class MotionTracker:
    def __init__(self, smoothing_window=15):
        self.motion_history = deque([0] * smoothing_window, maxlen=smoothing_window)
        self.smoothing_window = smoothing_window
    
    def get_smoothed_motion(self, current_motion):
        self.motion_history.append(current_motion)
        return sum(self.motion_history) / self.smoothing_window

def create_roi_mask(frame_shape):
    """Create a mask that focuses on the center of the frame where game action usually happens"""
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define the central region (ignore edges where stats usually appear)
    margin_x = int(width * 0.15)  # 15% margin from left and right
    margin_y = int(height * 0.2)  # 20% margin from top and bottom
    
    # Create a rectangle mask for the central region
    mask[margin_y:height-margin_y, margin_x:width-margin_x] = 255
    
    return mask

# Create a global motion tracker
motion_tracker = MotionTracker()

def detect_motion(prev_frame, current_frame, threshold=35, debug=False):
    # Convert frames to HSV color space
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    current_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    
    # Create ROI mask
    roi_mask = create_roi_mask(prev_frame.shape)
    
    # Calculate absolute difference for each channel
    h_diff = cv2.absdiff(prev_hsv[:,:,0], current_hsv[:,:,0])
    s_diff = cv2.absdiff(prev_hsv[:,:,1], current_hsv[:,:,1])
    v_diff = cv2.absdiff(prev_hsv[:,:,2], current_hsv[:,:,2])
    
    # Combine differences with adjusted weights (focus heavily on value/brightness)
    combined_diff = (h_diff * 0.1 + s_diff * 0.1 + v_diff * 0.8).astype(np.uint8)
    
    # Apply strong Gaussian blur to reduce noise
    combined_diff = cv2.GaussianBlur(combined_diff, (7, 7), 0)
    
    # Thresholding with higher value
    _, thresh = cv2.threshold(combined_diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply ROI mask
    thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
    
    # Apply morphological operations to remove small noise and connect nearby motion
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Remove small noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Connect nearby motion
    
    # Find contours to analyze motion regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_contour_area = (prev_frame.shape[0] * prev_frame.shape[1]) * 0.005  # 0.5% of frame size (reduced from 1%)
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Calculate motion percentage (relative to ROI area)
    roi_area = np.sum(roi_mask == 255)
    motion_score = sum(cv2.contourArea(cnt) for cnt in significant_contours)
    motion_percentage = (motion_score / roi_area) * 100
    
    # Get smoothed motion percentage
    smoothed_percentage = motion_tracker.get_smoothed_motion(motion_percentage)
    
    # Debug visualization
    if debug:
        debug_frame = np.zeros_like(prev_frame)
        
        # Draw motion regions in red
        debug_frame[:,:,2] = thresh
        
        # Draw significant contours in blue
        cv2.drawContours(debug_frame, significant_contours, -1, (255,0,0), 2)
        
        # Draw ROI rectangle in green
        height, width = prev_frame.shape[:2]
        margin_x = int(width * 0.15)
        margin_y = int(height * 0.2)
        cv2.rectangle(debug_frame, 
                     (margin_x, margin_y), 
                     (width-margin_x, height-margin_y), 
                     (0, 255, 0), 2)
        
        # Add text showing motion percentage
        cv2.putText(debug_frame, 
                   f"Motion: {motion_percentage:.1f}% (Smoothed: {smoothed_percentage:.1f}%)", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (0, 255, 0), 
                   2)
        
        cv2.imshow("Motion Detection Debug", debug_frame)
        cv2.waitKey(1)
    
    # Return both instant and smoothed motion detection
    return {
        'motion_detected': smoothed_percentage > 15.0,  # Lowered from 25% to 15%
        'significant_motion': smoothed_percentage > 5.0,  # Lower threshold to maintain recording
        'motion_percentage': smoothed_percentage
    }
