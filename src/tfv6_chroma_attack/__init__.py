"""White-box adversarial patch against TransFuser v6 (`tfv6_visiononly`).

Unlike `yolo_chroma_attack` / `clip_chroma_attack`, which attack an EXTERNAL
detector and hope for transfer, this package attacks the driving policy's own
differentiable heads (target-speed distribution, waypoints, CenterNet BEV
detection).

tfv6 does not consume a single 1280x720 FOV-90 frame: it consumes a 6-camera
360-degree surround composite of shape (384, 2304). The frames captured for the
YOLO/CLIP attacks are therefore geometrically incompatible and a new dataset has
to be captured through tfv6's exact rig — that is what `capture_tfv6.py` does.
`build_quads.py` then derives the patch quad per frame by differencing a clean
render against a patched one.
"""
