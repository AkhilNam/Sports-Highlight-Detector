# Real-Time Sports Highlights Detector

A Python application that automatically detects and extracts highlights from sports videos using OpenCV and advanced motion detection algorithms. Perfect for basketball, soccer, and other fast-paced sports.

## Features

- 🎥 **Video Processing**: Process pre-recorded videos to extract highlights
- 🏀 **Smart Motion Detection**: Advanced motion analysis with ROI masking and motion smoothing
- ⏱️ **Intelligent Highlight Timing**: 
  - Pre-roll capture (1.5s before motion starts)
  - Post-motion recording (2.0s after motion stops)
  - Minimum highlight duration (2.0s)
  - Maximum highlight duration (15.0s)
- 🎯 **Configurable Parameters**: Adjust motion thresholds, durations, and detection sensitivity
- 🔍 **Debug Mode**: Visualize motion detection and processing in real-time
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

## Usage

### Processing a Video File
```bash
python detector.py --mode video --file path/to/your/video.mp4
```

### Debug Mode (with motion visualization)
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
```

### Controls
- Press 'q' to quit the application
- Press 'r' to replay the last highlight
- In debug mode, motion detection visualization will be shown in a separate window

## How It Works

1. **Frame Processing**: The application processes video frames at a configurable sample rate
2. **Motion Detection**: Uses HSV color space analysis with ROI masking to detect significant motion
3. **Motion Smoothing**: Applies temporal smoothing to prevent false positives
4. **Highlight Detection**: 
   - Starts recording when significant motion is detected
   - Includes pre-roll frames for context
   - Continues recording during motion
   - Adds post-motion frames to complete the action
5. **Clip Saving**: Automatically saves highlights when motion ends or max duration is reached

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
- See `requirements.txt` for complete list

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 