import cv2
import argparse
import time
import os
from motion_detection import detect_motion

# Setup output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def process_frames(capture, mode="video"):
    prev_frame = None
    frame_count = 0
    sample_rate = 5  # Process every 5th frame

    if mode == "video":
        success, frame = capture.read()
    else:
        success, frame = True, next(capture)

    while success:
        if frame_count % sample_rate == 0:
            if prev_frame is not None:
                motion_detected = detect_motion(prev_frame, frame)
                if motion_detected:
                    log(f"⚡ Motion detected at frame {frame_count}")
                    save_path = os.path.join(OUTPUT_DIR, f"highlight_{frame_count}.jpg")
                    cv2.imwrite(save_path, frame)
                    log(f"✅ Saved highlight: {save_path}")

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

    process_frames(capture, mode=args.mode)

if __name__ == "__main__":
    main()
