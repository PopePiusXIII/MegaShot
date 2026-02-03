"""
PROMPT FOR COPILOT:
Create a robust VideoAnimationEngine class for a video editing app.

REQUIREMENTS:
1. Coordinate System: Use normalized coordinates (0.0 to 1.0) instead of pixels to ensure 
   scaling across different UI preview sizes.
2. Data Structure: Manage 'keyframes' as a list of dictionaries containing:
   {'timestamp': float, 'x': float, 'y': float, 'color': tuple, 'brush_size': int}
3. Interpolation Logic: Implement a method `get_state_at_time(time)` that uses 
   Linear Interpolation (LERP) to calculate the exact (x, y) and color between two keyframes.
4. Frame Rendering: Create a `render_overlay` method that takes an OpenCV frame and a 
   timestamp, then draws the interpolated points/lines using cv2.circle or cv2.line.
5. Optimization: Ensure the engine doesn't reload the video for every frame; 
   it should be optimized for a 'scrubbing' timeline.

LIBRARIES: cv2, numpy
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
from golf_ball_trajectory import GolfBallTrajectory

class VideoAnimationEngine:
    """
    Manages keyframe-based animated overlays on video frames using normalized coordinates.
    Keyframe format: {'timestamp': float, 'x': float, 'y': float, 'color': (B,G,R), 'brush_size': int}
      - x,y are normalized [0.0..1.0]
      - color is (B,G,R) 0-255 ints
      - brush_size is int 0..100 representing percent of min(frame_w,frame_h)
    """

    def __init__(self, video_path: Optional[str] = None, capture: Optional[cv2.VideoCapture] = None):
        self.keyframes: List[Dict] = []
        self.cap = capture if capture is not None else (cv2.VideoCapture(video_path) if video_path else None)
        self._last_frame_time: Optional[float] = None
        self._last_frame = None

    # -----------------------
    # Keyframe management
    # -----------------------
    def add_keyframe(self, kf: Dict):
        """Add keyframe dict and keep list sorted by timestamp."""
        if not all(k in kf for k in ("timestamp", "x", "y", "color", "brush_size")):
            raise ValueError("Keyframe must contain timestamp, x, y, color, brush_size")
        self.keyframes.append(kf.copy())
        self.keyframes.sort(key=lambda k: k["timestamp"])

    def remove_keyframe_at(self, timestamp: float, tolerance: float = 1e-6):
        """Remove keyframes matching timestamp within tolerance."""
        self.keyframes = [k for k in self.keyframes if abs(k["timestamp"] - timestamp) > tolerance]

    def clear_keyframes(self):
        self.keyframes.clear()

    # -----------------------
    # Interpolation (LERP)
    # -----------------------
    def get_state_at_time(self, t: float) -> Optional[Dict]:
        """
        Return interpolated state at time t using linear interpolation.
        Returns dict: {'timestamp': t, 'x': float, 'y': float, 'color': (B,G,R), 'brush_size': int}
        If no keyframes: returns None
        """
        if not self.keyframes:
            return None

        # before first or after last
        if t <= self.keyframes[0]["timestamp"]:
            k = self.keyframes[0]
            return {"timestamp": t, "x": k["x"], "y": k["y"], "color": tuple(k["color"]), "brush_size": int(k["brush_size"])}
        if t >= self.keyframes[-1]["timestamp"]:
            k = self.keyframes[-1]
            return {"timestamp": t, "x": k["x"], "y": k["y"], "color": tuple(k["color"]), "brush_size": int(k["brush_size"])}

        # find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            a = self.keyframes[i]
            b = self.keyframes[i + 1]
            if a["timestamp"] <= t <= b["timestamp"]:
                span = b["timestamp"] - a["timestamp"]
                if span == 0:
                    alpha = 0.0
                else:
                    alpha = (t - a["timestamp"]) / span
                x = float(a["x"] + alpha * (b["x"] - a["x"]))
                y = float(a["y"] + alpha * (b["y"] - a["y"]))
                col_a = np.array(a["color"], dtype=float)
                col_b = np.array(b["color"], dtype=float)
                color = tuple(np.clip((col_a + alpha * (col_b - col_a)).round().astype(int), 0, 255))
                brush = int(round(a["brush_size"] + alpha * (b["brush_size"] - a["brush_size"])))
                return {"timestamp": t, "x": x, "y": y, "color": color, "brush_size": brush}
        return None  # fallback

    # -----------------------
    # Video/frame helpers
    # -----------------------
    def set_video(self, video_path: str):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(video_path)
        self._last_frame_time = None
        self._last_frame = None

    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Seek to time (seconds) and return frame. Uses simple caching to avoid redundant reads
        when scrubbing near same time.
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        # reuse cached frame if same timestamp requested
        if self._last_frame_time is not None and abs(self._last_frame_time - timestamp) < 1e-3:
            return self._last_frame.copy() if self._last_frame is not None else None

        # Seek by milliseconds then read
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ret, frame = self.cap.read()
        if not ret:
            return None
        self._last_frame_time = timestamp
        self._last_frame = frame.copy()
        return frame

    # -----------------------
    # Rendering overlay
    # -----------------------
    def _brush_pixels(self, brush_size_int: int, frame_shape: Tuple[int, int, int]) -> int:
        """
        Convert brush_size int (0..100) to pixel radius relative to min(frame_w,frame_h).
        Ensures radius >= 1.
        """
        h, w = frame_shape[:2]
        radius = max(1, int(round((brush_size_int / 100.0) * min(w, h))))
        return radius

    def render_overlay(self, frame: np.ndarray, timestamp: float, show_trail: bool = True, fps: int = 240) -> np.ndarray:
        """
        Draws overlay onto a copy of frame for given timestamp.
        - Draws current interpolated point as a filled circle.
        - Optionally draws a trail by sampling intermediate states from first keyframe time to timestamp.
        - fps: frames per second for sampling (default 240, typical iPhone slow-mo)
        Returns annotated frame (does not modify input).
        """
        out = frame.copy()
        h, w = out.shape[:2]
        state = self.get_state_at_time(timestamp)
        if state is None:
            return out

        # Draw trail (sampled)
        if show_trail and self.keyframes:
            start_t = self.keyframes[0]["timestamp"]
            duration = max(timestamp - start_t, 0.01)
            samples = int(fps * duration)
            if timestamp > start_t and samples > 1:
                times = np.linspace(start_t, timestamp, num=samples)
                prev_pt = None
                prev_col = None
                prev_br = None
                for tt in times:
                    s = self.get_state_at_time(float(tt))
                    if s is None:
                        continue
                    px = int(round(s["x"] * w))
                    py = int(round(s["y"] * h))
                    col = tuple(int(c) for c in s["color"])
                    br = self._brush_pixels(s["brush_size"], out.shape)
                    if prev_pt is not None:
                        cv2.line(out, prev_pt, (px, py), color=col, thickness=max(1, int(round((br + prev_br) / 2))))
                    prev_pt = (px, py)
                    prev_col = col
                    prev_br = br

        # Draw current point
        px = int(round(state["x"] * w))
        py = int(round(state["y"] * h))
        color = tuple(int(c) for c in state["color"])
        radius = self._brush_pixels(state["brush_size"], out.shape)
        cv2.circle(out, (px, py), radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

        return out

    def save_animation_video(self, output_video_path: str, duration: float, fps: float = None, show_trail: bool = True, samples: int = 80):
        """
        Save an animation video with overlays to the specified path.
        - output_video_path: output file path
        - duration: total duration in seconds
        - fps: frames per second (if None, uses video fps)
        - show_trail: whether to show the trail
        - samples: number of samples for the trail
        """
        if self.cap is None or not self.cap.isOpened():
            print("Error: No video loaded or cannot open video.")
            return
        # Get video properties
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps is None:
            fps = video_fps if video_fps > 0 else 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        num_frames = int(duration * fps)
        for i in range(num_frames):
            t = i / fps
            frame = self.get_frame_at_time(t)
            if frame is None:
                continue
            overlay = self.render_overlay(frame, t, show_trail=show_trail)
            out_vid.write(overlay)
        out_vid.release()
        print(f"Saved animation video as {output_video_path}")


def project_to_keyframe(x, y, z, width=1920, height=1080, fov_deg=65, cam_pos=( -5, 0, 0 ), cam_euler=(10, 0, 0)):
    """
    Project real-world (x, y, z) to normalized keyframe coordinates using a pinhole camera model.
    Camera at cam_pos (x, y, z), with euler angles (pitch, yaw, roll) in degrees.
    Default: camera is 5m behind ball, pitched down 10 deg.
    """
    # Camera extrinsics
    cam_x, cam_y, cam_z = cam_pos
    pitch, yaw, roll = np.radians(cam_euler)
    # Translate ball position to camera coordinates
    pt = np.array([x - cam_x, y - cam_y, z - cam_z])
    # Rotation matrix (ZYX order: roll, pitch, yaw)
    Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    Rz = np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]])
    R = Rz @ Ry @ Rx
    pt_cam = R @ pt
    x_c, y_c, z_c = pt_cam
    # Pinhole projection
    f = (width / 2) / np.tan(np.radians(fov_deg) / 2)
    if x_c <= 0.1:
        x_c = 0.1
    x_img = f * y_c / x_c + width / 2
    y_img = f * (-z_c) / x_c + height / 2
    x_norm = np.clip(x_img / width, 0, 1)
    y_norm = np.clip(y_img / height, 0, 1)
    return x_norm, y_norm


if __name__ == "__main__":
    # Example usage: Golf ball trajectory animation using GolfBallTrajectory class
    engine = VideoAnimationEngine("golfVideo.MOV")

    # Generate keyframes from golf ball trajectory
    T_max = 3.0  # seconds
    traj = GolfBallTrajectory(
        t_max=T_max,
        x0=0, y0=0, z0=0,
        v0=70,  # m/s
        launch_angle_deg=15,
        azimuth_deg=15,
        side_spin_rpm=-2000,
        back_spin_rpm=3000,
        dt=0.01
    )
    traj.sim()
    for i in range(len(traj.t)):
        t = traj.t[i]
        x, y, z = traj.x[i], traj.y[i], traj.z[i]
        norm_x, norm_y = project_to_keyframe(x, y, z)
        color = (255, 0, 0) 
        brush_size = 1.0
        engine.add_keyframe({'timestamp': t, 'x': norm_x, 'y': norm_y, 'color': color, 'brush_size': brush_size})

    # Save a video showing the golf ball trajectory animation over time using the new method
    output_video_path = "golfVideoOut.mp4"
    duration = T_max
    engine.save_animation_video(output_video_path, duration, show_trail=True, samples=80)
