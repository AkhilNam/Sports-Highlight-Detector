# Changes Log

## Summary of Motion Detection Evolution

### Problem 1: Over-Detection
**Initial Issue:**
- Every frame was being detected as a highlight
- System couldn't distinguish between game action and UI/graphics
- No filtering for small movements

**Solution:**
- Added motion percentage threshold (5%)
- Added cooldown period between highlights
- Added minimum highlight duration

### Problem 2: UI/Stats Triggering
**Issue:**
- Black backgrounds in stats screens causing false positives
- Grayscale detection was too sensitive to UI changes
- Edges of frame (where stats appear) causing triggers

**Solution:**
- Switched from grayscale to HSV color space
- Added ROI (Region of Interest) masking
- Increased margins to ignore UI elements

### Problem 3: Commentator Scenes
**Issue:**
- Small movements (like commentators talking) triggering detection
- Scattered motion being counted as significant
- No size requirement for motion areas

**Final Solution:**
- Implemented contour detection for motion regions
- Required minimum size for motion areas (1% of frame)
- Increased motion threshold to 25%
- Added morphological operations to clean up noise
- Weighted brightness changes more heavily (70%)
- Added strong Gaussian blur (7x7)

### Current Parameters
```python
# ROI Margins
margin_x = 15%  # of frame width
margin_y = 20%  # of frame height

# Motion Detection
motion_threshold = 25%  # of ROI area
min_contour_area = 1%  # of frame size
gaussian_blur = (7,7)
channel_weights = (0.1, 0.2, 0.7)  # hue, saturation, value

# Highlight Management
cooldown = 2.0 seconds
min_duration = 1.0 seconds
```

## Motion Detection Improvements (2024-03-21)

### Problem Identified
- Original code was detecting too many highlights
- Every frame with motion was being saved as a highlight
- No consideration for highlight duration or cooldown periods

### Changes Made

#### 1. Motion Detection (`motion_detection.py`)
```python
# Old version
return motion_score > 50000  # Fixed threshold

# New version
frame_size = prev_frame.shape[0] * prev_frame.shape[1]
motion_percentage = (motion_score / frame_size) * 100
return motion_percentage > 5.0
```
**Why this change?**
- Fixed threshold (50000) didn't work well across different video resolutions
- New version calculates motion as a percentage of frame size
- More robust across different video qualities and resolutions
- 5% threshold means significant motion must cover at least 5% of the frame

#### 2. Highlight Management (`detector.py`)
```python
# Added parameters
HIGHLIGHT_COOLDOWN = 2.0      # seconds between highlights
MIN_HIGHLIGHT_DURATION = 1.0  # minimum duration of a highlight
FRAME_RATE = 30              # frames per second
```

**Why these changes?**
- Prevents rapid-fire highlights from the same event
- Ensures highlights are meaningful moments, not just single frames
- Added cooldown period to prevent duplicate highlights
- Minimum duration ensures highlights are actual events, not just blips

### Results
- Significantly reduced number of false positives
- Highlights are now more meaningful and spaced out
- System is more robust across different video qualities

### Parameters to Tune
These values can be adjusted based on testing:
- Motion percentage threshold (currently 5%)
- Highlight cooldown (currently 2.0 seconds)
- Minimum highlight duration (currently 1.0 seconds)
- Frame rate (currently 30 fps)

### Next Steps
1. Test with different sports footage
2. Adjust parameters based on sport type
3. Consider adding sport-specific thresholds
4. Implement adaptive thresholding for different lighting conditions

## Color-Based Motion Detection (2024-03-21)

### Problem Identified
- Grayscale conversion was causing issues with black backgrounds
- Motion detection was triggering on stats/UI elements instead of game action
- Needed better visualization of what was being detected

### Changes Made

#### 1. HSV Color Space Motion Detection
```python
# Convert to HSV color space
prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
current_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

# Calculate differences for each channel
h_diff = cv2.absdiff(prev_hsv[:,:,0], current_hsv[:,:,1])
s_diff = cv2.absdiff(prev_hsv[:,:,1], current_hsv[:,:,1])
v_diff = cv2.absdiff(prev_hsv[:,:,2], current_hsv[:,:,2])

# Weighted combination
combined_diff = (h_diff * 0.2 + s_diff * 0.3 + v_diff * 0.5)
```

**Why this change?**
- HSV color space better handles different lighting conditions
- Separate channel analysis helps distinguish real motion from UI changes
- Weighted combination prioritizes value (brightness) changes
- More robust against black backgrounds and UI elements

#### 2. Debug Visualization
```python
if debug:
    debug_frame = np.zeros_like(prev_frame)
    debug_frame[:,:,2] = thresh  # Red channel for motion
    cv2.imshow("Motion Detection Debug", debug_frame)
```

**Why this change?**
- Helps visualize exactly what motion is being detected
- Makes it easier to tune parameters
- Useful for debugging false positives/negatives

### How to Test
Run the detector with debug mode:
```bash
python detector.py --mode video --file your_video.mp4 --debug
```

This will show:
1. Original video
2. Motion detection visualization (red areas show detected motion)
3. Console logs of motion detection events

### Next Steps
1. Test with different types of sports footage
2. Adjust HSV channel weights based on testing
3. Consider adding region-of-interest filtering
4. Implement adaptive thresholds based on scene content 

### Added
- Initial release of the Sports Highlights Detector
- Video processing mode for pre-recorded videos
- Advanced motion detection with ROI masking
- Intelligent highlight timing with pre-roll and post-motion capture
- Multiple video codec support for compatibility
- Debug mode with motion visualization
- Highlight preview functionality

### Features
- **Motion Detection**
  - HSV color space analysis
  - ROI masking to focus on center of frame
  - Motion smoothing over 15 frames
  - Configurable motion thresholds
  - Minimum contour area filtering

- **Highlight Timing**
  - Pre-roll capture (1.5s before motion)
  - Post-motion recording (2.0s after motion)
  - Minimum highlight duration (2.0s)
  - Maximum highlight duration (15.0s)
  - Cooldown period between highlights (2.0s)

- **Video Processing**
  - Support for multiple video codecs
  - Automatic codec fallback
  - Frame rate preservation
  - Output directory management

- **User Interface**
  - Real-time motion visualization in debug mode
  - Highlight duration display
  - Recording status indicator
  - Keyboard controls (q to quit, r to replay)

### Technical Details
- Python 3.8+ compatibility
- OpenCV for video processing
- NumPy for numerical operations
- Efficient frame processing with configurable sample rate
- Robust error handling and logging

### Known Issues
- None reported in initial release

### Future Improvements
- Live stream processing support
- Customizable ROI regions
- Advanced motion analysis algorithms
- Support for multiple sports types
- Batch processing capabilities 