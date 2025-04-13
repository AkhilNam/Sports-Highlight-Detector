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

## Sport-Specific Detection Improvements (2024-03-22)

### Problem Identified
- Generic motion detection not effective for different sports
- Ball tracking in soccer was unreliable
- Basketball detection needed sport-specific features
- Debug visualization was insufficient

### Changes Made

#### 1. Sport-Specific Detection Architecture
```python
class SportDetector:
    def __init__(self, sport_type="basketball", confidence_threshold=0.5):
        self.sport_type = sport_type.lower()
        self.confidence_threshold = confidence_threshold
        self.model = YOLO('yolov8n.pt')
        
        # Sport-specific parameters
        if self.sport_type == "basketball":
            self.classes = [0]  # Only track people
        else:  # soccer
            self.classes = [0, 32]  # Track people and sports ball
```

**Why this change?**
- Separate detection logic for each sport
- Customizable parameters per sport
- More accurate action detection
- Better performance for specific sports

#### 2. Enhanced Ball Tracking
```python
class BallTracker:
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.positions = deque(maxlen=max_history)
        self.velocities = deque(maxlen=max_history-1)
        self.last_position = None
        self.predicted_position = None
```

**Why this change?**
- Predicts ball position when detection is lost
- Maintains ball trajectory history
- Calculates ball velocity for action detection
- Smoother tracking between frames

#### 3. Improved Debug Visualization
```python
# Draw ball and trajectory
if ball:
    x1, y1, x2, y2, conf = ball
    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    ball_center = ((x1 + x2)/2, (y1 + y2)/2)
    cv2.circle(debug_frame, (int(ball_center[0]), int(ball_center[1])), 5, (0, 0, 255), -1)
    
    # Draw ball trajectory
    if len(self.ball_tracker.positions) > 1:
        points = np.array(self.ball_tracker.positions, dtype=np.int32)
        cv2.polylines(debug_frame, [points], False, (0, 255, 255), 2)
```

**Why this change?**
- Visualizes ball trajectory
- Shows ball velocity vector
- Displays action type and statistics
- Better debugging capabilities

### Sport-Specific Parameters
```python
# Basketball Parameters
basketball_params = {
    'min_players_for_action': 2,
    'min_player_confidence': 0.5,
    'court_region_threshold': 0.3,
    'fast_break_threshold': 0.4,
}

# Soccer Parameters
soccer_params = {
    'min_players_for_action': 3,
    'min_player_confidence': 0.5,
    'min_ball_confidence': 0.3,
    'goal_region_threshold': 0.2,
    'ball_speed_threshold': 0.2,
    'min_players_near_ball': 2,
    'ball_tracking_radius': 50,
}
```

### Results
- More accurate highlight detection for each sport
- Better ball tracking in soccer
- Improved action classification
- Enhanced debugging capabilities

### Next Steps
1. Add support for more sports types
2. Implement sport-specific ROI calibration
3. Add audio analysis for crowd reactions
4. Improve ball tracking accuracy
5. Add support for multiple camera angles 

## Performance and Sensitivity Improvements (2024-04-13)

### Problems Identified
1. Processing Speed:
   - System was processing every frame, causing slow performance
   - Full resolution processing was unnecessary
   - No frame skipping optimization

2. Soccer Detection Sensitivity:
   - Too few highlights being detected
   - High confidence thresholds causing missed actions
   - Limited types of action detection
   - Required too many players for action detection

### Changes Made

#### 1. Processing Speed Optimization
```python
# Frame Processing Optimization
sample_rate = 3  # Process every 3rd frame
small_frame = cv2.resize(frame, (640, 360))  # Process at lower resolution
```

**Why these changes?**
- 3x faster processing by skipping frames
- Lower resolution processing maintains accuracy while improving speed
- More efficient resource usage
- Better real-time performance

#### 2. Soccer Detection Sensitivity Improvements
```python
# Soccer-specific parameters
self.min_players = 3  # Reduced from 6 to 3
self.action_threshold = 0.3  # Lowered from 0.5
self.ball_confidence_threshold = 0.2  # New parameter
self.player_confidence_threshold = 0.3  # New parameter
```

**New Detection Methods:**
```python
def _detect_ball_movement(self, detections):
    # Detects any significant ball movement
    return movement > 10  # Threshold for significant movement

def _detect_player_clustering(self, detections):
    # Detects when multiple players are near the ball
    return nearby_players >= 2  # At least 2 players near ball
```

**Why these changes?**
- Lower confidence thresholds catch more action
- New ball movement detection catches fast plays
- Player clustering detection identifies build-up play
- More comprehensive action detection
- Better highlight coverage

### Updated Parameters
```python
# Highlight Management
HIGHLIGHT_COOLDOWN = 5.0      # Increased from 2.0
MIN_HIGHLIGHT_DURATION = 20.0 # Increased from 2.0
MAX_HIGHLIGHT_DURATION = 45.0 # Increased from 15.0
PRE_ROLL_SECONDS = 3.0        # Increased from 1.5
POST_MOTION_SECONDS = 5.0     # Increased from 2.0

# Soccer Detection
min_players = 3              # Reduced from 6
action_threshold = 0.3       # Reduced from 0.5
ball_confidence = 0.2        # New parameter
player_confidence = 0.3      # New parameter
```

### Results
- 3x faster processing speed
- More comprehensive highlight detection
- Better capture of different types of soccer action
- Longer, more meaningful highlights
- Improved action context with pre/post roll

### Next Steps
1. Implement adaptive frame skipping based on action intensity
2. Add crowd noise analysis for additional action detection
3. Develop sport-specific ROI calibration
4. Add support for multiple camera angles
5. Implement highlight quality scoring

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

## Highlight Merging Improvements (2024-04-13)

### Problem Identified
- Small gaps between potential highlights were causing unnecessary splits
- Related actions were being separated into multiple clips
- No consideration for continuous play sequences

### Changes Made

#### 1. Highlight Merging Logic (`detector.py`)
```python
# Added merge threshold parameter
MERGE_GAP_THRESHOLD = 2.0  # seconds - merge highlights if gap is smaller than this

# Added merge condition
should_merge = (is_highlight_active and 
                (current_time - last_action_time) <= MERGE_GAP_THRESHOLD and
                sport_info['has_action'])
```

**Why these changes?**
- Prevents unnecessary splitting of continuous action
- Maintains context between related plays
- Creates more natural highlight sequences
- Better captures the flow of the game

#### 2. Action Tracking
```python
# Track last action time
last_action_time = 0  # Track when the last action was detected

# Update last action time when action is detected
if sport_info['has_action']:
    last_action_time = current_time
```

**Why these changes?**
- Enables gap detection between actions
- Helps determine when to merge highlights
- Maintains action continuity
- Better handles brief pauses in play

### Updated Parameters
```python
# Highlight Management
HIGHLIGHT_COOLDOWN = 3.0      # Reduced from 5.0
MIN_HIGHLIGHT_DURATION = 15.0 # Reduced from 20.0
MAX_HIGHLIGHT_DURATION = 45.0 # Kept same
PRE_ROLL_SECONDS = 5.0        # Increased from 3.0
POST_ACTION_SECONDS = 8.0     # Increased from 5.0
MERGE_GAP_THRESHOLD = 2.0     # New parameter
```

### Results
- More natural highlight sequences
- Better capture of continuous play
- Reduced number of unnecessary clip splits
- Improved highlight context and flow
- More meaningful highlight durations

### Next Steps
1. Implement adaptive merge threshold based on sport type
2. Add highlight quality scoring for merged clips
3. Consider crowd noise analysis for merge decisions
4. Add support for multi-camera merge synchronization
5. Implement highlight summary generation