# Real-Time Sports Highlights Detector

A Python application that automatically detects and extracts highlights from sports videos using OpenCV, YOLOv8 object detection, and sport-specific action analysis. Perfect for basketball and soccer highlights.

## Features

- 🎥 **Video Processing**: Process pre-recorded videos to extract highlights
- 🏀 **Sport-Specific Detection**: 
  - Basketball: Fast breaks, scoring opportunities, player movements
  - Soccer: Ball tracking, shots on goal, counter attacks, player clustering
- 🎯 **Advanced Object Detection**: YOLOv8-based detection of players and ball
- ⚡ **Optimized Performance**: 
  - Frame skipping for faster processing
  - Lower resolution detection for efficiency
  - 3x faster processing speed
- ⏱️ **Intelligent Highlight Timing**: 
  - Pre-roll capture (5.0s before action starts)
  - Post-action recording (8.0s after action ends)
  - Minimum highlight duration (15.0s)
  - Maximum highlight duration (45.0s)
  - Smart highlight merging for continuous action
- 🎯 **Configurable Parameters**: Adjust sport-specific thresholds and timing
- 🔍 **Debug Mode**: Visualize sport-specific detection in real-time
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
# For basketball
python detector.py --mode video --file path/to/your/video.mp4 --sport basketball

# For soccer
python detector.py --mode video --file path/to/your/video.mp4 --sport soccer
```

### Debug Mode (with sport-specific visualization)
```bash
python detector.py --mode video --file path/to/your/video.mp4 --sport soccer --debug
```

### Parameters
The following parameters can be adjusted in `detector.py`:

```python
# Timing Parameters
HIGHLIGHT_COOLDOWN = 3.0      # seconds between highlights
MIN_HIGHLIGHT_DURATION = 15.0 # minimum duration of a highlight
MAX_HIGHLIGHT_DURATION = 45.0 # maximum duration of a highlight
PRE_ROLL_SECONDS = 5.0        # seconds to keep before action starts
POST_ACTION_SECONDS = 8.0     # seconds to keep recording after action ends
MERGE_GAP_THRESHOLD = 2.0     # seconds - merge highlights if gap is smaller than this

# Soccer-Specific Parameters
min_players = 2              # minimum players for action
action_threshold = 0.2       # action detection threshold
ball_confidence = 0.15       # ball detection confidence
player_confidence = 0.2      # player detection confidence
```

### Controls
- Press 'q' to quit the application
- Press 'r' to replay the last highlight
- In debug mode:
  - Sport-specific detection visualization
  - Action type display
  - Real-time statistics
  - Ball trajectory visualization
  - Player clustering visualization

## How It Works

1. **Frame Processing**: The application processes video frames at a configurable sample rate (every 3rd frame)
2. **Sport-Specific Detection**: 
   - Basketball: Analyzes player movements, court regions, and scoring opportunities
   - Soccer: Tracks ball movement, player interactions, goal opportunities, and player clustering
3. **Action Analysis**: Identifies significant plays based on sport-specific criteria
4. **Highlight Detection**: 
   - Starts recording when significant action is detected
   - Includes pre-roll frames for context
   - Continues recording during active play
   - Merges nearby highlights with small gaps (< 2.0s)
   - Adds post-action frames to complete the play
5. **Clip Saving**: Automatically saves highlights when action ends or max duration is reached

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