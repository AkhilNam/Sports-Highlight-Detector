# Real-Time Sports Highlights Detector

A Python application that automatically detects and extracts highlights from sports videos using OpenCV, YOLOv8 object detection, and advanced motion analysis. Perfect for basketball, soccer, and other fast-paced sports.

## Features

- 🎥 **Video Processing**: Process pre-recorded videos to extract highlights
- 🏀 **Smart Motion Detection**: Advanced motion analysis with ROI masking and motion smoothing
- 🎯 **Object Detection**: YOLOv8-based detection of players and ball
- ⏱️ **Intelligent Highlight Timing**: 
  - Pre-roll capture (1.5s before motion starts)
  - Post-motion recording (2.0s after motion stops)
  - Minimum highlight duration (2.0s)
  - Maximum highlight duration (15.0s)
- 🎯 **Configurable Parameters**: Adjust motion thresholds, object detection sensitivity, and timing
- 🔍 **Debug Mode**: Visualize motion detection and object tracking in real-time
- 📹 **Multiple Codec Support**: Automatically tries different video codecs for compatibility
- 🔄 **Highlight Preview**: Instantly replay the last detected highlight

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sports-highlight-detector.git
cd sports-highlight-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The first run will automatically download the YOLOv8 model weights.

## Usage

### Processing a Video File
```bash
python detector.py --mode video --file path/to/your/video.mp4
```

### Debug Mode (with motion and object detection visualization)
```bash
python detector.py --mode video --file path/to/your/video.mp4 --debug
```

### Parameters
The following parameters can be adjusted in `detector.py`:

```python
# Timing Parameters
HIGHLIGHT_COOLDOWN = 2.0      # seconds between highlights
MIN_HIGHLIGHT_DURATION = 2.0  # minimum duration of a highlight
MAX_HIGHLIGHT_DURATION = 15.0 # maximum duration of a highlight
PRE_ROLL_SECONDS = 1.5        # seconds to keep before motion starts
POST_MOTION_SECONDS = 2.0     # seconds to keep recording after motion stops

# Motion Detection Parameters
MIN_MOTION_TO_KEEP_RECORDING = 8.0  # minimum motion percentage to maintain recording

# Object Detection Parameters
OBJECT_DETECTION_SAMPLE_RATE = 5     # Process every Nth frame for object detection
MIN_PLAYERS_FOR_ACTION = 2           # Minimum number of players for significant action
MIN_BALL_CONFIDENCE = 0.5            # Minimum confidence for ball detection
MIN_PLAYER_CONFIDENCE = 0.5          # Minimum confidence for player detection
```

### Controls
- Press 'q' to quit the application
- Press 'r' to replay the last highlight
- In debug mode:
  - Motion detection visualization shown in one window
  - Object detection visualization shown in another window
  - Real-time statistics displayed on screen

## How It Works

1. **Frame Processing**: The application processes video frames at a configurable sample rate
2. **Motion Detection**: Uses HSV color space analysis with ROI masking to detect significant motion
3. **Object Detection**: YOLOv8 detects players and ball, tracking their positions and interactions
4. **Action Analysis**: Combines motion and object detection to identify significant plays
5. **Highlight Detection**: 
   - Starts recording when significant motion or action is detected
   - Includes pre-roll frames for context
   - Continues recording during active play
   - Adds post-motion frames to complete the action
6. **Clip Saving**: Automatically saves highlights when action ends or max duration is reached

## Output

Highlights are saved in the `output/clips` directory with timestamps in the filename:
```
output/
  └── clips/
      ├── highlight_1234567890.mp4
      ├── highlight_1234567891.mp4
      └── ...
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLOv8)
- See `requirements.txt` for complete list

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 