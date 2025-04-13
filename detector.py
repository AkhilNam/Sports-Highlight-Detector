import cv2
import argparse
import time
import os
from motion_detection import detect_motion

# Setup output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Highlight detection parameters
HIGHLIGHT_COOLDOWN = 2.0  # seconds between highlights
MIN_HIGHLIGHT_DURATION = 1.0  # minimum duration of a highlight
FRAME_RATE = 30  # assuming 30fps, adjust based on your video

def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def get_video_capture(source):
    return cv2.VideoCapture(source)

def get_screen_capture():
    import mss
    import numpy as np
    sct = mss.mss()
    monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}
    while True:
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        yield frame

def process_frames(capture, mode="video", debug=False):
    prev_frame = None
    frame_count = 0
    sample_rate = 5  # Process every 5th frame
    last_highlight_time = 0
    highlight_start_time = 0
    is_highlight_active = False
    highlight_frames = []

    if mode == "video":
        success, frame = capture.read()
    else:
        success, frame = True, next(capture)

    while success:
        current_time = frame_count / FRAME_RATE

        if frame_count % sample_rate == 0:
            if prev_frame is not None:
                motion_detected = detect_motion(prev_frame, frame, debug=debug)
                
                if debug:
                    log(f"Frame {frame_count}: Motion detected: {motion_detected}")
                
                if motion_detected and (current_time - last_highlight_time) >= HIGHLIGHT_COOLDOWN:
                    if not is_highlight_active:
                        is_highlight_active = True
                        highlight_start_time = current_time
                        highlight_frames = []
                        log(f"🎬 Starting new highlight at {current_time:.2f}s")
                    
                    highlight_frames.append(frame.copy())
                
                elif is_highlight_active:
                    if (current_time - highlight_start_time) >= MIN_HIGHLIGHT_DURATION:
                        # Save the highlight
                        save_path = os.path.join(OUTPUT_DIR, f"highlight_{int(time.time())}.jpg")
                        cv2.imwrite(save_path, highlight_frames[-1])
                        log(f"✅ Saved highlight: {save_path}")
                        last_highlight_time = current_time
                    
                    is_highlight_active = False
                    highlight_frames = []

            prev_frame = frame.copy()

            # Display (optional, can remove for headless mode)
            cv2.imshow("Sports Highlights Detector", frame)

        frame_count += 1

        if mode == "video":
            success, frame = capture.read()
        else:
            frame = next(capture)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            log("🛑 Quitting detection.")
            break

    if mode == "video":
        capture.release()

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Sports Highlights Detector")
    parser.add_argument("--mode", choices=["video", "live"], required=True, help="Choose mode: 'video' or 'live'")
    parser.add_argument("--file", type=str, help="Path to video file (if mode is 'video')")
    parser.add_argument("--stream-url", type=str, help="Stream URL (if mode is 'live')")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to visualize motion detection")

    args = parser.parse_args()

    if args.mode == "video":
        if not args.file:
            log("❌ Please provide a video file path with --file")
            return
        capture = get_video_capture(args.file)
    elif args.mode == "live":
        if args.stream_url:
            capture = get_video_capture(args.stream_url)
        else:
            capture = get_screen_capture()

    process_frames(capture, mode=args.mode, debug=args.debug)

if __name__ == "__main__":
    main()
