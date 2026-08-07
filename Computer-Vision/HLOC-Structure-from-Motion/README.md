# HLOC Structure-from-Motion and Visual Localization

A computer vision project implementing a Structure-from-Motion and visual localization pipeline using Hierarchical Localization (HLOC).

The project reconstructs a scene from multiple images and estimates the camera pose of a query image relative to the reconstructed 3D model.

## Pipeline

The notebook implements the following workflow:

```text
Reference Images
      ↓
DISK Feature Extraction
      ↓
Exhaustive Image Pairing
      ↓
LightGlue Feature Matching
      ↓
Structure-from-Motion Reconstruction
      ↓
3D Scene Model
      ↓
Query Image
      ↓
Feature Extraction and Matching
      ↓
Camera Pose Estimation
```

## Feature Extraction and Matching

The HLOC configuration used in the notebook is:

```text
Feature extractor: DISK
Matcher: LightGlue
Maximum keypoints: 5000
Maximum image size: 1600
```

Reference-image pairs are generated exhaustively before feature matching.

## 3D Reconstruction

The matched image features are passed to the HLOC reconstruction pipeline to create a Structure-from-Motion model.

The recorded notebook run used 13 reference images and produced:

```text
Registered images: 13
3D points: 6833
Observations: 34343
Mean track length: 5.03
Mean observations per image: 2641.77
Mean reprojection error: 1.18 px
```

The reconstructed point cloud and camera poses are visualized in 3D.

The notebook also visualizes local feature observations and correspondences between selected images.

## Visual Localization

A separate query image is localized inside the reconstructed scene.

The localization pipeline:

1. Extracts DISK features from the query image.
2. Matches the query against registered reference images using LightGlue.
3. Infers the query camera model using PyCOLMAP.
4. Builds 2D-to-3D correspondences.
5. Estimates the camera pose using geometric verification and RANSAC.
6. Refines camera parameters.
7. Visualizes the estimated camera pose and inlier 3D points.

The localization configuration uses a RANSAC maximum error of `12` and enables refinement of focal length and additional camera parameters.

In the recorded run, localization found:

```text
1456 inlier correspondences
2381 total correspondences
```

## Technologies

- Python
- HLOC
- DISK
- LightGlue
- PyCOLMAP
- Structure-from-Motion
- Visual Localization
- NumPy
- Plotly
- Google Colab

## Project File

[`HLOC.ipynb`](./HLOC.ipynb) contains the complete feature extraction, matching, reconstruction, localization and visualization pipeline.

## Notes

The notebook uses pretrained local feature extraction and matching models through HLOC. It does not train DISK or LightGlue from scratch.
