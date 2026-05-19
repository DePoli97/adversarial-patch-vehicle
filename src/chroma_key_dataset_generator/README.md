# Chroma-key dataset generator

Goal: build a CARLA-rendered dataset to train an adversarial patch against
a target perception model (YOLO, tfv6 detection head, etc).

## Pipeline overview

1. **Marker texture** (`build_yellow_tga.py`)
   Generate a TGA where the rear surface of the target vehicle is painted
   solid yellow (chroma-key marker). Reimport the TGA on the vehicle's
   glass/body material in Unreal Editor.

2. **Scenario sweep** (`generate_dataset.py`, TODO)
   Loop over (town, weather, sun-altitude, leader-distance, seed) and run a
   short two-vehicle scenario. Save the follower's front-camera frames and a
   sidecar JSON per frame with environment metadata.

3. **Quad extraction** (`extract_quad.py`, TODO)
   For each frame, isolate the yellow region with an HSV mask, fit the four
   corners with `cv2.approxPolyDP`, save the corners as JSON.

4. **Training time** (separate notebook, TODO)
   During patch training, the dataset gives (image, quad). The patch tensor
   is differentiably warped onto the quad region (replacing the yellow
   placeholder) before passing through the target model.

The yellow marker is **only a placeholder for coordinates** — it never
appears in the training input the model sees. The chroma-key approach
avoids having to re-render the dataset every time the patch changes.

## Files

- `build_yellow_tga.py` — generates the marker TGAs. Replaces the
  rear-window region of the target vehicle's shared glass texture with a
  solid yellow block whose position matches the existing canvas-trick
  layout from `src/patch_on_surface/build_rear_window_tga.py`.
- (more coming)
