# Hand Gesture Recognition

Real-time hand gesture recognition using **MediaPipe** for hand landmark detection and a **Random Forest** classifier (scikit-learn) for gesture classification.

## Overview

This project implements a complete pipeline to collect, process, and classify hand gestures from a webcam feed. It uses [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) to extract 21 3D landmarks per hand, then applies custom normalization (rotation, scaling, feature engineering) to make recognition invariant to hand position, size, and orientation.

The system recognizes **5 gestures** (right hand):

| Gesture    | Description  |
| ---------- | ------------ |
| CloseFist  | Closed fist  |
| OpenHand   | Open hand    |
| ThumbsUp   | Thumbs up    |
| IndexUp    | Index finger |
| OkSign     | OK sign      |

## Pipeline

``` text
                +-----------+
                |  Webcam   |
                +-----+-----+
                      |
                      v
            +-------------------+
            |   1. Capture      |   Extract 21 3D landmarks per frame
            |   (MediaPipe)     |   via MediaPipe Hands
            +--------+----------+
                     |
                     v
            +-------------------+
            |  2. Normalize     |   Scale, rotate to canonical pose,
            |                   |   compute geometric features
            +--------+----------+
                     |
                     v
            +-------------------+
            |  3. Fuse          |   Merge per-person files into
            |                   |   per-gesture datasets
            +--------+----------+
                     |
                     v
            +-------------------+
            |  4. Train         |   Train RandomForest classifier
            |  (scikit-learn)   |   on 78-dim feature vectors
            +--------+----------+
                     |
                     v
            +-------------------+
            |  5. Predict       |   Real-time classification
            |                   |   from webcam feed
            +-------------------+
```

### Step-by-step

1. **Capture** (`gesture-capture`) — Opens the webcam, detects the hand using MediaPipe, and saves the 21 world-space 3D landmarks to a JSON file each time you press SPACE.

2. **Normalize** (`gesture-normalize`) — For each sample, normalizes the landmarks to be invariant to hand position, scale, and rotation. Also computes 15 additional geometric features (finger angles, finger lengths, inter-finger distances). See [Normalization](docs/normalization.md) for details.

3. **Fuse** (`gesture-fuse`) — Merges all per-person normalized files into a single file per gesture (e.g. all `CloseFist_*.json` into `CloseFist_normalized.json`).

4. **Train** (`gesture-train`) — Loads the fused data, builds a 78-dimensional feature vector per sample (21 landmarks x 3 coords + 15 features), trains a Random Forest classifier (200 trees), and saves the model + scaler as a `.pkl` file.

5. **Predict** (`gesture-predict`) — Opens the webcam, processes each frame through MediaPipe + normalization + classifier, and displays the predicted gesture with confidence score in real time.

## Installation

**Prerequisites:** Python >= 3.12, a webcam, and [uv](https://docs.astral.sh/uv/).

``` bash
git clone https://github.com/<your-username>/hand-gesture-recognition.git
cd hand-gesture-recognition
uv sync
```

## Usage

### Quick start (use the pre-trained model)

``` bash
# Run real-time prediction with the included model
uv run gesture-predict
```

Press **ESC** to quit.

### Full pipeline (collect your own data)

``` bash
# 1. Capture gesture samples (repeat for each gesture + person)
uv run gesture-capture --gesture CloseFist --name Alice
uv run gesture-capture --gesture OpenHand --name Alice
# ... repeat for all 5 gestures and all contributors

# 2. Normalize raw data
uv run gesture-normalize

# 3. Fuse normalized data by gesture
uv run gesture-fuse

# 4. Train the model (displays confusion matrix)
uv run gesture-train

# 5. Run real-time prediction
uv run gesture-predict
```

### Visualization

``` bash
# Visualize a hand sample in 3D (matplotlib interactive plot)
uv run gesture-visualize data/normalized/CloseFist_William_normalized.json --index 0
```

### CLI reference

Each command supports `--help` for full usage details.

| Command              | Description                                  | Key options                             |
| -------------------- | -------------------------------------------- | --------------------------------------- |
| `gesture-capture`    | Record hand landmarks from webcam            | `--gesture`, `--name`, `--output-dir`   |
| `gesture-normalize`  | Normalize raw JSON files                     | `--input-dir`, `--output-dir`           |
| `gesture-fuse`       | Merge normalized files by gesture            | `--gestures`, `--input-dir`             |
| `gesture-train`      | Train the RandomForest model                 | `--data-dir`, `--model-dir`             |
| `gesture-predict`    | Real-time gesture recognition                | `--model`                               |
| `gesture-visualize`  | 3D plot of hand landmarks                    | `json_file`, `--index`                  |

### Keyboard shortcuts

| Key       | Context   | Action                    |
| --------- | --------- | ------------------------- |
| **SPACE** | Capture   | Save current hand pose    |
| **ESC**   | Any       | Quit the application      |

## Project structure

``` text
.
├── src/hand_gesture/          # Python package
│   ├── __init__.py
│   ├── capture.py             # Webcam data collection
│   ├── normalize.py           # Landmark normalization + feature extraction
│   ├── fuse.py                # Merge per-person files into per-gesture datasets
│   ├── train.py               # Model training & evaluation
│   ├── predict.py             # Real-time gesture prediction
│   └── visualize.py           # 3D landmark visualization (matplotlib)
├── data/
│   ├── raw/                   # Raw captured JSON (21 landmarks x 3D coords)
│   ├── normalized/            # Normalized JSON (21 landmarks + 15 features)
│   └── fused/                 # One merged file per gesture
├── models/                    # Trained model + scaler (.pkl)
├── docs/                      # Documentation
│   ├── normalization.md       # Normalization algorithm details
│   ├── data-format.md         # JSON data format specification
│   └── adding-gestures.md     # How to add new gestures
├── pyproject.toml
├── LICENSE
└── README.md
```

## Feature vector

Each hand sample is represented as a **78-dimensional** vector:

| Dimensions | Content                                                                 |
| ---------- | ----------------------------------------------------------------------- |
| 0-62       | 21 landmarks x 3 coordinates (x, y, z), rotation-normalized             |
| 63-67      | 5 finger curl angles (thumb, index, middle, ring, pinky)                |
| 68-72      | 5 finger lengths (tip-to-base distance)                                 |
| 73         | Thumb-index angle                                                       |
| 74         | Thumb-index distance                                                    |
| 75-77      | Thumb-index relative position (delta x, y, z)                           |

See [docs/normalization.md](docs/normalization.md) for the full normalization algorithm.

## Dataset

The included dataset was collected by **4 contributors** (Jules, Martin, Matthias, William), right hand only. Each person captured multiple sessions per gesture, totaling several hundred samples per class. Data is stored as JSON files containing MediaPipe [world landmarks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#world_landmarks) (3D coordinates in meters, relative to the hand's geometric center).

## Model

- **Algorithm:** Random Forest (200 trees, no max depth)
- **Preprocessing:** StandardScaler on the 78-dim feature vector
- **Train/test split:** 80/20, random_state=42
- **Output:** `models/hand_gesture_rf.pkl` (pickled tuple of `(RandomForestClassifier, StandardScaler)`)

## Documentation

- [Normalization algorithm](docs/normalization.md) — Detailed explanation of the scale/rotation normalization and feature engineering
- [Data format specification](docs/data-format.md) — JSON schema for raw, normalized, and fused data files
- [Adding new gestures](docs/adding-gestures.md) — Step-by-step guide to extend the system with custom gestures

## Requirements

| Dependency      | Version    | Purpose                          |
| --------------- | ---------- | -------------------------------- |
| Python          | >= 3.12    | Runtime                          |
| MediaPipe       | 0.10.14    | Hand landmark detection          |
| OpenCV          | 4.10.0     | Webcam capture & display         |
| NumPy           | 1.26.4     | Numerical operations             |
| scikit-learn    | >= 1.4.0   | RandomForest classifier          |
| seaborn         | >= 0.13.0  | Confusion matrix visualization   |
| joblib          | >= 1.3.0   | Model serialization              |

## License

[MIT](LICENSE)
