import cv2
import argparse
import time
import os
import subprocess
import json
from collections import deque
from motion_detection import detect_motion
from sport_detection import SportDetector

# Setup output directory
OUTPUT_DIR = "output"
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# Highlight detection parameters
HIGHLIGHT_COOLDOWN = 3.0  # seconds between highlights
MIN_HIGHLIGHT_DURATION = 15.0  # minimum duration of a highlight
MAX_HIGHLIGHT_DURATION = 45.0  # maximum duration of a highlight
FRAME_RATE = 30  # assuming 30fps, adjust based on your video
PRE_ROLL_SECONDS = 5.0  # seconds to keep before motion starts
POST_MOTION_SECONDS = 8.0  # seconds to keep recording after motion stops
MERGE_GAP_THRESHOLD = 2.0  # seconds - merge highlights if gap is smaller than this

def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def save_highlight_clip(frames, output_path, fps=30):
    """Save a sequence of frames as a video clip"""
    if not frames:
        log("❌ No frames to save")
        return False
        
    try:
        height, width = frames[0].shape[:2]
        
        # Try different codecs in order of preference
        codecs = [
            ('avc1', '.mp4'),
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi')
        ]
        
        for codec, ext in codecs:
            try:
                # Update output path with correct extension
                output_file = os.path.splitext(output_path)[0] + ext
                
                # Create VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    log(f"⚠️ Failed to initialize VideoWriter with codec {codec}")
                    continue
                
                # Write frames
                for frame in frames:
                    out.write(frame)
                
                out.release()
                log(f"✅ Successfully saved clip with codec {codec}")
                return True
                
            except Exception as e:
                log(f"⚠️ Error with codec {codec}: {str(e)}")
                continue
                
        log("❌ Failed to save clip with any codec")
        return False
        
    except Exception as e:
        log(f"❌ Error saving clip: {str(e)}")
        return False

def get_stream_url(url):
    """Get the direct stream URL using yt-dlp"""
    try:
        # Run yt-dlp to get stream info
        cmd = [
            'yt-dlp',
            '-f', 'best',  # Get best quality
            '-g',  # Get direct URL
            url
        ]
        direct_url = subprocess.check_output(cmd).decode('utf-8').strip()
        return direct_url
    except subprocess.CalledProcessError as e:
        log(f"❌ Error getting stream URL: {e}")
        return None

def get_video_capture(source, is_stream=False):
    """Get video capture object for file or stream"""
    if is_stream:
        log("🔄 Getting stream URL...")
        direct_url = get_stream_url(source)
        if not direct_url:
            return None
        log("✅ Got stream URL, connecting...")
        cap = cv2.VideoCapture(direct_url)
        
        # Set buffer size to minimize delay
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        return cap
    else:
        return cv2.VideoCapture(source)

def get_screen_capture():
    """Capture screen region"""
    import mss
    import numpy as np
    sct = mss.mss()
    monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}
    while True:
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        yield frame

def process_frames(capture, mode="video", sport="basketball", debug=False):
    prev_frame = None
    frame_count = 0
    sample_rate = 3  # Process every 3rd frame instead of every frame
    last_highlight_time = 0
    highlight_start_time = 0
    is_highlight_active = False
    highlight_frames = []
    last_action_time = 0  # Track when the last action was detected
    
    # Initialize sport detector
    sport_detector = SportDetector(sport_type=sport)
    
    # Pre-roll buffer
    pre_roll_size = int(PRE_ROLL_SECONDS * FRAME_RATE)
    frame_buffer = deque(maxlen=pre_roll_size)
    
    # For preview window
    cv2.namedWindow("Sports Highlights Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sports Highlights Detector", 1280, 720)
    
    # For clips preview
    latest_clip = None
    show_clip = False
    clip_frame_idx = 0

    if mode == "video":
        success, frame = capture.read()
    else:
        success, frame = True, next(capture)

    while success:
        current_time = frame_count / FRAME_RATE
        
        # Always add frame to pre-roll buffer
        if frame is not None:
            frame_buffer.append(frame.copy())

        # Only process every sample_rate frames
        if frame_count % sample_rate == 0:
            if prev_frame is not None:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (640, 360))
                
                # Run sport-specific detection
                sport_info = sport_detector.detect_action(small_frame, debug=debug)
                
                if debug:
                    log(f"Frame {frame_count}: Action: {sport_info['action_type'] if sport_info['has_action'] else 'None'}")
                
                # Check if we should merge with previous highlight
                should_merge = (is_highlight_active and 
                              (current_time - last_action_time) <= MERGE_GAP_THRESHOLD and
                              sport_info['has_action'])
                
                # Start highlight if we have significant action
                if (sport_info['has_action'] and 
                    (not is_highlight_active or should_merge) and 
                    (current_time - last_highlight_time) >= HIGHLIGHT_COOLDOWN):
                    
                    if not is_highlight_active:
                        is_highlight_active = True
                        highlight_start_time = current_time - PRE_ROLL_SECONDS
                        highlight_frames = list(frame_buffer)
                        log(f"🎬 Starting new highlight at {highlight_start_time:.2f}s (with {len(frame_buffer)} pre-roll frames)")
                    else:
                        log(f"🔄 Merging with previous highlight (gap: {current_time - last_action_time:.2f}s)")
                
                if is_highlight_active:
                    highlight_frames.append(frame.copy())
                    if debug:
                        log(f"📼 Collected frame {len(highlight_frames)} for current highlight")
                    
                    # Update last action time if we have action
                    if sport_info['has_action']:
                        last_action_time = current_time
                    
                    # Check if we've hit max duration
                    current_duration = current_time - highlight_start_time
                    if current_duration >= MAX_HIGHLIGHT_DURATION:
                        log(f"⏰ Max duration reached ({current_duration:.1f}s)")
                        # Save the highlight video clip
                        timestamp = int(time.time())
                        log(f"💾 Attempting to save highlight with {len(highlight_frames)} frames...")
                        clip_path = os.path.join(CLIPS_DIR, f"highlight_{timestamp}.mp4")
                        if save_highlight_clip(highlight_frames, clip_path, FRAME_RATE):
                            latest_clip = highlight_frames
                            show_clip = True
                            clip_frame_idx = 0
                        last_highlight_time = current_time
                        is_highlight_active = False
                        highlight_frames = []
                        continue
                    
                    # Only stop recording if action has ended and gap is too large
                    if not sport_info['has_action'] and (current_time - last_action_time) > MERGE_GAP_THRESHOLD:
                        if (current_time - highlight_start_time) >= MIN_HIGHLIGHT_DURATION:
                            # Save the highlight video clip
                            timestamp = int(time.time())
                            log(f"💾 Attempting to save highlight with {len(highlight_frames)} frames...")
                            clip_path = os.path.join(CLIPS_DIR, f"highlight_{timestamp}.mp4")
                            if save_highlight_clip(highlight_frames, clip_path, FRAME_RATE):
                                latest_clip = highlight_frames
                                show_clip = True
                                clip_frame_idx = 0
                            last_highlight_time = current_time
                        else:
                            log(f"⏳ Highlight too short ({current_time - highlight_start_time:.1f}s < {MIN_HIGHLIGHT_DURATION}s)")
                        
                        is_highlight_active = False
                        highlight_frames = []

            prev_frame = frame.copy()

            # Show either the live feed or the latest highlight clip
            if show_clip and latest_clip:
                display_frame = latest_clip[clip_frame_idx].copy()
                clip_frame_idx = (clip_frame_idx + 1) % len(latest_clip)
                if clip_frame_idx == 0:  # Clip finished playing
                    show_clip = False
                cv2.putText(display_frame, "REPLAY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                display_frame = frame.copy()
                if is_highlight_active:
                    duration = current_time - highlight_start_time
                    cv2.putText(display_frame, "🔴 RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Duration: {duration:.1f}s / {MAX_HIGHLIGHT_DURATION}s", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if sport_info['has_action']:
                        cv2.putText(display_frame, f"Action: {sport_info['action_type']}", (10, 110), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Sports Highlights Detector", display_frame)

        frame_count += 1

        if mode == "video":
            success, frame = capture.read()
        else:
            try:
                frame = next(capture)
            except StopIteration:
                log("⚠️ Stream ended, attempting to reconnect...")
                if isinstance(capture, cv2.VideoCapture):
                    capture.release()
                    time.sleep(5)  # Wait before reconnecting
                    capture = get_video_capture(args.stream_url, is_stream=True)
                    if capture is None:
                        break
                    success, frame = capture.read()
                else:
                    break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log("🛑 Quitting detection.")
            break
        elif key == ord('r'):  # Press 'r' to replay last highlight
            if latest_clip:
                show_clip = True
                clip_frame_idx = 0

    if isinstance(capture, cv2.VideoCapture):
        capture.release()

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Sports Highlights Detector")
    parser.add_argument("--mode", choices=["video", "live"], required=True, help="Choose mode: 'video' or 'live'")
    parser.add_argument("--file", type=str, help="Path to video file (if mode is 'video')")
    parser.add_argument("--stream-url", type=str, help="YouTube/Twitch stream URL (if mode is 'live')")
    parser.add_argument("--sport", choices=["basketball", "soccer"], default="basketball", help="Sport type for detection")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to visualize detection")

    args = parser.parse_args()

    if args.mode == "video":
        if not args.file:
            log("❌ Please provide a video file path with --file")
            return
        capture = get_video_capture(args.file)
    elif args.mode == "live":
        if args.stream_url:
            log("🎥 Connecting to stream...")
            capture = get_video_capture(args.stream_url, is_stream=True)
            if capture is None:
                log("❌ Failed to connect to stream")
                return
        else:
            capture = get_screen_capture()

    process_frames(capture, mode=args.mode, sport=args.sport, debug=args.debug)

if __name__ == "__main__":
    main()
