# VideoAnimationEngine

This document explains how `VideoAnimationEngine` works, how to use it, and how the key concepts fit together. The engine lives in [src/VideoAnimatorEngine.py](src/VideoAnimatorEngine.py) and provides a simple, timeline-driven way to draw animated overlays on top of video frames using normalized coordinates.

## Goals and design

- **Normalized coordinates**: All keyframe positions are stored as `x` and `y` in the range `[0.0, 1.0]`. This makes overlays resolution-independent and keeps UI scaling consistent.
- **Keyframe-driven animation**: The engine uses a list of keyframes (timestamped points) and interpolates between them using linear interpolation (LERP).
- **Efficient scrubbing**: A basic frame cache prevents redundant reads when the timeline is scrubbed repeatedly near the same time.
- **OpenCV-friendly**: Rendering uses OpenCV primitives (`cv2.circle`, `cv2.line`) and operates directly on frames.

## Core data model

A **keyframe** is a dictionary with this shape:

```
{
    "timestamp": float,   # seconds
    "x": float,           # normalized [0.0, 1.0]
    "y": float,           # normalized [0.0, 1.0]
    "color": (B, G, R),   # ints 0..255
    "brush_size": int     # 0..100, percent of min(frame_w, frame_h)
}
```

Keyframes are stored in a list at `self.keyframes`. When you add a keyframe, the list is kept sorted by timestamp.

## Class overview

### `VideoAnimationEngine.__init__(video_path=None, capture=None)`

- Loads a video through `cv2.VideoCapture` if `video_path` is provided.
- You can pass an existing `cv2.VideoCapture` via `capture`.
- Initializes a small cache for the last accessed frame.

### `add_keyframe(kf)`

- Validates required fields.
- Adds the keyframe and sorts the list by `timestamp`.

### `remove_keyframe_at(timestamp, tolerance=1e-6)`

- Removes all keyframes that match `timestamp` within `tolerance`.

### `clear_keyframes()`

- Empties the keyframe list.

### `get_state_at_time(t)`

Returns the interpolated overlay state at time `t`.

- If `t` is before the first keyframe or after the last keyframe, it returns the first or last keyframe value.
- If `t` falls between two keyframes, the method performs linear interpolation:

For each property:

- **Position**: $x(t) = x_0 + \alpha (x_1 - x_0)$, same for $y$.
- **Color**: interpolated per channel, then rounded and clamped to `[0, 255]`.
- **Brush size**: interpolated and rounded to an integer.

Where:

$$
\alpha = \frac{t - t_0}{t_1 - t_0}
$$

If no keyframes exist, it returns `None`.

### `set_video(video_path)`

- Releases any existing capture.
- Loads the new video.
- Clears the frame cache.

### `get_frame_at_time(timestamp)`

- Uses `cv2.CAP_PROP_POS_MSEC` to seek to a specific time.
- Returns a frame at that time, or `None` if the read fails.
- If called repeatedly with the same timestamp (within 1 ms), it returns a cached frame to reduce decoding work.

### `render_overlay(frame, timestamp, show_trail=True, fps=240)`

Draws the overlay for a given `timestamp` on a copy of `frame`.

- **Current point**: drawn as a filled circle at the interpolated state.
- **Trail (optional)**: the engine samples intermediate states from the first keyframe to the current `timestamp` and connects them with lines.
- `fps` controls the sampling density for the trail. Higher values produce a smoother trail, but cost more CPU.

### `render_full_trajectory_overlay(frame, color=(0,255,255), brush_size=1.0)`

Draws the full trajectory using all keyframes, independent of time.

- Connects keyframe points with a polyline.
- Draws a circle at each keyframe location.
- Uses the provided `color` and `brush_size` for the entire trajectory.

### `find_golf_ball_in_keyframe(...)`

Detects a golf ball in a single frame using a simple color mask and contour filtering.

- Converts to HSV and thresholds by color range.
- Cleans up the mask using morphology.
- Filters contours by area and circularity.
- Returns the best candidate near the center of the frame.

This is a convenience utility and can be used to generate or validate keyframes automatically.

### `save_animation_video(output_video_path, duration, fps=None, show_trail=True, samples=80)`

Renders and writes a new video with the overlay applied.

- `duration` is the total output length in seconds.
- If `fps` is not given, the engine uses the source video FPS.
- Each output frame is rendered by calling `render_overlay` and written to `cv2.VideoWriter`.

## Example usage

Below is a minimal example that generates keyframes from a `GolfBallTrajectory` and writes the output video:

```python
from golf_ball_trajectory import GolfBallTrajectory
from VideoAnimatorEngine import VideoAnimationEngine, project_to_keyframe

engine = VideoAnimationEngine("tests/data/videos/golfVideo.MOV")

traj = GolfBallTrajectory(
    t_max=3.0,
    x0=0, y0=0, z0=0,
    v0=70,
    launch_angle_deg=15,
    azimuth_deg=15,
    side_spin_rpm=-2000,
    back_spin_rpm=3000,
    dt=0.01,
)
traj.sim()

for i in range(len(traj.t)):
    t = traj.t[i]
    x, y, z = traj.x[i], traj.y[i], traj.z[i]
    norm_x, norm_y = project_to_keyframe(x, y, z)
    engine.add_keyframe({
        "timestamp": t,
        "x": norm_x,
        "y": norm_y,
        "color": (255, 0, 0),
        "brush_size": 1,
    })

engine.save_animation_video("golfVideoOut.mp4", duration=3.0, show_trail=True)
```

## Coordinate system and projection

The engine assumes **normalized coordinates** for all drawing operations:

- `x = 0.0` is the left edge, `x = 1.0` is the right edge
- `y = 0.0` is the top edge, `y = 1.0` is the bottom edge

The helper `project_to_keyframe(x, y, z, ...)` converts 3D world coordinates to normalized 2D coordinates using a pinhole camera model and Euler angles. This is useful when your source data is in real-world units.

## Performance notes

- Scrubbing performance depends on `get_frame_at_time`. The engine avoids re-reading the same timestamp but still seeks for other timestamps. If you need high-performance scrubbing, consider caching nearby frames or keeping a decoded buffer.
- Trails are sampled at a fixed `fps`. For long durations, this can become expensive. Reduce `fps` or limit the trail to a sliding window if needed.

## Common pitfalls

- **Incorrect color order**: OpenCV uses BGR. If you use RGB, colors will look swapped.
- **Normalized values**: Ensure `x` and `y` are in `[0.0, 1.0]`. Out-of-range values can draw off-screen.
- **Brush size**: `brush_size` is a percentage of the minimum frame dimension, not pixels.
- **Video paths**: Use the correct data path: `tests/data/videos/golfVideo.MOV`.

## Suggested extensions

- Add `add_trajectory_keyframes(...)` to generate keyframes directly from `GolfBallTrajectory` parameters.
- Add a `CameraModel` config object to keep projection parameters in one place.
- Add interpolation options (e.g., cubic) for smoother paths.
- Add an option to draw trails using alpha blending for fade-out effects.

## Where to look

- Engine class: [src/VideoAnimatorEngine.py](src/VideoAnimatorEngine.py)
- Trajectory model: [src/golf_ball_trajectory.py](src/golf_ball_trajectory.py)
- Example data: [tests/data/videos](tests/data/videos)
