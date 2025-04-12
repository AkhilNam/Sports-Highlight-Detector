
# 🏟️ Real-Time Sports Highlights Detector

> Detect highlights from **live sports streams** or **full game recordings** automatically.  
> Clips big moments, creates highlight reels, and optionally sends real-time alerts.

---

## 🚀 Project Overview

This project is a computer vision-powered system that watches sports games — either **live** or from **recorded footage** — and automatically detects highlights such as goals, dunks, or touchdowns. It saves exciting moments as clips, and can even compile full highlight reels.

**Modes Supported:**
- 🎥 **Full game video mode** — Process a game recording and extract all highlights.
- 📡 **Live stream mode** — Detect highlights in real time from YouTube / Twitch / or screen-capture.

---

## 🎯 Features

- ✅ Real-time live stream analysis
- ✅ Full-game video processing
- ✅ Motion-based highlight detection
- ✅ AI-based action recognition (optional, Phase 2)
- ✅ Audio analysis for crowd and commentator excitement (optional, Phase 2)
- ✅ Auto-save highlight clips
- ✅ Auto-generate highlight reels
- ✅ Optional: Live dashboard to view moments in real time
- ✅ Optional: SMS / Email notifications for detected highlights

---

## 🧩 Tech Stack

| Component | Tools |
|-----------|-------|
| Frame Capture | OpenCV, ffmpeg, yt-dlp, mss (for screen cap) |
| Motion Detection | Frame differencing, histogram analysis |
| Action Recognition (Phase 2) | PyTorchVideo, SlowFast Networks, YOLOv8 |
| Audio Analysis (Optional) | librosa, ffmpeg-python |
| Clip Saving | ffmpeg-python, MoviePy |
| Dashboard (Optional) | Flask / FastAPI + React.js |
| Notifications (Optional) | Twilio / SMTP |
| Deployment (Optional) | AWS EC2 / GCP VM |

---

## 🗓️ Project Timeline

| Week | Milestone |
|------|-----------|
| Week 1 | Dual-mode pipeline: live stream & video file + motion-based clip detection |
| Week 2 | Add smart detection: AI models, optional audio analysis |
| Week 3 | Auto-highlight reel generation, optional dashboard |
| Week 4 (Stretch) | Real-time alerts, cloud deployment, multi-game support |

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/sports-highlights-detector.git
cd sports-highlights-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Usage

**Video File Mode:**
```bash
python detector.py --mode video --file path/to/game.mp4
```

**Live Stream Mode:**
```bash
python detector.py --mode live --stream-url "https://youtube.com/stream"
```

Clips will be saved automatically in the `output/` directory!

---

## 🌟 Future Improvements

- Fine-tune action recognition models for specific sports
- Improve audio-visual fusion for better highlight accuracy
- Real-time social media auto-posting
- Multi-stream support (monitor multiple games simultaneously)
- Deploy to cloud for 24/7 monitoring

---

## 🤖 Contributing

If you have ideas, feel free to open an issue or submit a pull request!  
We welcome contributions that make the project better.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🎥 Demo

_(Coming soon)_  
Stay tuned for a full video demo showcasing live highlight detection!
