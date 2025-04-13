import cv2
import numpy as np

# HSV Weight Configurations
HSV_WEIGHTS = {
    'default': (0.1, 0.2, 0.7),     # Current: Focus on brightness
    'color': (0.3, 0.4, 0.3),       # Good for jersey detection
    'motion': (0.1, 0.1, 0.8),      # Best for fast movement
    'balanced': (0.33, 0.33, 0.34)  # Equal weights
}

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

def detect_motion(prev_frame, current_frame, threshold=35, debug=False, hsv_preset='default'):
    """
    Detect motion between frames using HSV color space analysis.
    
    Args:
        prev_frame: Previous video frame
        current_frame: Current video frame
        threshold: Motion detection threshold (default: 35)
        debug: Whether to show debug visualization (default: False)
        hsv_preset: HSV weight preset to use (default: 'default')
                   Options: 'default', 'color', 'motion', 'balanced'
    """
    # Get HSV weights
    h_weight, s_weight, v_weight = HSV_WEIGHTS.get(hsv_preset, HSV_WEIGHTS['default'])
    
    # Convert frames to HSV color space
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    current_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    
    # Create ROI mask
    roi_mask = create_roi_mask(prev_frame.shape)
    
    # Calculate absolute difference for each channel
    h_diff = cv2.absdiff(prev_hsv[:,:,0], current_hsv[:,:,0])
    s_diff = cv2.absdiff(prev_hsv[:,:,1], current_hsv[:,:,1])
    v_diff = cv2.absdiff(prev_hsv[:,:,2], current_hsv[:,:,2])
    
    # Combine differences with configured weights
    combined_diff = (h_diff * h_weight + s_diff * s_weight + v_diff * v_weight).astype(np.uint8)
    
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
    min_contour_area = (prev_frame.shape[0] * prev_frame.shape[1]) * 0.01  # 1% of frame size
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Calculate motion percentage (relative to ROI area)
    roi_area = np.sum(roi_mask == 255)
    motion_score = sum(cv2.contourArea(cnt) for cnt in significant_contours)
    motion_percentage = (motion_score / roi_area) * 100
    
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
        
        # Add text showing motion percentage and HSV preset
        cv2.putText(debug_frame, 
                   f"Motion: {motion_percentage:.1f}% ({hsv_preset})", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (0, 255, 0), 
                   2)
        
        cv2.imshow("Motion Detection Debug", debug_frame)
        cv2.waitKey(1)
    
    # Much stricter threshold - require significant motion
    return (motion_percentage > 25.0  # Increased from 15% to 25%
            and len(significant_contours) >= 1)  # Must have at least one large motion region
