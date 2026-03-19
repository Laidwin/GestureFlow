# Data Format Specification

This document describes the JSON format used at each stage of the pipeline.

## Raw data (`data/raw/`)

**Filename pattern:** `{Gesture}_{Person}.json` (e.g. `CloseFist_William.json`)

Each file contains samples captured by one person for one gesture.

``` json
{
  "result": [
    {
      "CloseFist_William_1": {
        "WRIST": { "x": -0.0456, "y": 0.0812, "z": 0.0023 },
        "THUMB_CMC": { "x": -0.0312, "y": 0.0567, "z": 0.0189 },
        "THUMB_MCP": { "x": -0.0198, "y": 0.0345, "z": 0.0234 },
        ...
        "PINKY_TIP": { "x": 0.0234, "y": -0.0456, "z": -0.0123 }
      }
    },
    {
      "CloseFist_William_2": {
        ...
      }
    }
  ]
}
```

### Fields

| Field     | Type   | Description                                                                     |
| --------- | ------ | ------------------------------------------------------------------------------- |
| `result`  | array  | List of captured hand samples                                                   |
| Entry key | string | `{Gesture}_{Person}_{SampleNumber}` — unique identifier                         |
| Landmark  | object | `{ "x": float, "y": float, "z": float }` — MediaPipe world coordinates (meters) |

There are exactly **21 landmarks** per sample, matching the [MediaPipe hand landmark model](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

## Normalized data (`data/normalized/`)

**Filename pattern:** `{Gesture}_{Person}_normalized.json`

Same structure as raw, but with two changes:

1. The 21 landmark coordinates are **scale- and rotation-normalized** (see [normalization.md](normalization.md))
2. **15 additional scalar features** are appended to each sample

``` json
{
  "result": [
    {
      "CloseFist_William_1": {
        "WRIST": { "x": 0.0, "y": 0.0, "z": 0.0 },
        "THUMB_CMC": { "x": 0.123, "y": -0.045, "z": 0.067 },
        ...
        "PINKY_TIP": { "x": -0.234, "y": 0.123, "z": -0.089 },
        "THUMB_ANGLE": 145.23,
        "INDEX_ANGLE": 167.89,
        "MIDDLE_FINGER_ANGLE": 170.12,
        "RING_FINGER_ANGLE": 168.45,
        "PINKY_ANGLE": 155.67,
        "THUMB_LENGTH": 0.456,
        "INDEX_LENGTH": 0.523,
        "MIDDLE_FINGER_LENGTH": 0.534,
        "RING_FINGER_LENGTH": 0.498,
        "PINKY_LENGTH": 0.412,
        "THUMB_INDEX_ANGLE": 45.67,
        "THUMB_INDEX_DISTANCE": 0.234,
        "THUMB_INDEX_X": 0.123,
        "THUMB_INDEX_Y": -0.045,
        "THUMB_INDEX_Z": 0.012
      }
    }
  ]
}
```

### Extra features

| Feature                  | Type  | Unit    | Description                                |
| ------------------------ | ----- | ------- | ------------------------------------------ |
| `THUMB_ANGLE`            | float | degrees | Thumb curl angle at IP joint               |
| `INDEX_ANGLE`            | float | degrees | Index finger curl angle                    |
| `MIDDLE_FINGER_ANGLE`    | float | degrees | Middle finger curl angle                   |
| `RING_FINGER_ANGLE`      | float | degrees | Ring finger curl angle                     |
| `PINKY_ANGLE`            | float | degrees | Pinky curl angle                           |
| `THUMB_LENGTH`           | float | ratio   | Thumb tip-to-base distance (normalized)    |
| `INDEX_LENGTH`           | float | ratio   | Index tip-to-base distance (normalized)    |
| `MIDDLE_FINGER_LENGTH`   | float | ratio   | Middle tip-to-base distance (normalized)   |
| `RING_FINGER_LENGTH`     | float | ratio   | Ring tip-to-base distance (normalized)     |
| `PINKY_LENGTH`           | float | ratio   | Pinky tip-to-base distance (normalized)    |
| `THUMB_INDEX_ANGLE`      | float | degrees | Angle between thumb and index at thumb MCP |
| `THUMB_INDEX_DISTANCE`   | float | ratio   | Distance between thumb tip and index PIP   |
| `THUMB_INDEX_X`          | float | ratio   | X-axis separation (thumb-index)            |
| `THUMB_INDEX_Y`          | float | ratio   | Y-axis separation (thumb-index)            |
| `THUMB_INDEX_Z`          | float | ratio   | Z-axis separation (thumb-index)            |

## Fused data (`data/fused/`)

**Filename pattern:** `{Gesture}_normalized.json`

Same structure as normalized data, but all samples from all contributors are merged into a single file per gesture.

``` json
{
  "result": [
    { "CloseFist_Person1_1": { ... } },
    { "CloseFist_Person1_2": { ... } },
    { "CloseFist_Person2_1": { ... } },
    { "CloseFist_Person3_1": { ... } },
    ...
  ]
}
```

## Model file (`models/hand_gesture_rf.pkl`)

A pickled Python tuple containing:

``` python
(RandomForestClassifier, StandardScaler)
```

- **RandomForestClassifier** — Trained scikit-learn model (200 estimators)
- **StandardScaler** — Fitted scaler used to standardize the 78-dim feature vectors before prediction

Load with:

``` python
import joblib
model, scaler = joblib.load("models/hand_gesture_rf.pkl")
```
