# Golf Shot Tracer

This project provides a video-based golf shot tracer and trajectory simulation/visualization tool.

## Features
- Keyframe-based video overlay engine
- Realistic golf ball trajectory simulation (with drag, spin, Magnus effect)
- Pinhole camera projection for overlay
- Test scripts for trajectory and projection analysis

## Getting Started

### 1. Clone the repository
```
git clone <your-repo-url>
cd Shot Tracer
```

### 2. Install dependencies
It is recommended to use a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run test scripts
- To test the trajectory and projection math:
  ```
  python test_video_animator_engine.py
  ```
- To test the projection performance:
  ```
  python test_projection_performance.py
  ```

### 4. Run the video overlay
- Edit `VideoAnimatorEngine.py` to set your video file and parameters.
- Run the script to generate an output video with the shot tracer overlay.

## Notes
- Requires Python 3.8+
- For video overlay, you must have a compatible video file in the project directory.
- The code is designed for educational and research use.

## License
MIT
