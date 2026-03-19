# Normalization Algorithm

This document describes the normalization pipeline applied to raw MediaPipe hand landmarks before they are fed to the classifier.

## Goal

Raw MediaPipe world landmarks are 3D coordinates (in meters) relative to the hand's approximate geometric center. However, they still vary based on:

- **Hand size** — Different people have different hand sizes
- **Hand orientation** — The same gesture can be performed at different angles
- **Hand position** — Absolute coordinates shift based on where the hand is in space

The normalization step makes the feature vector **invariant** to all three factors, so the classifier only sees the gesture's intrinsic shape.

## Step 1: Scale normalization

All 21 landmark coordinates are divided by the **wrist-to-ring-finger-MCP distance** (Euclidean distance between landmarks #0 and #13). This removes the effect of hand size.

``` text
distance = ||RING_FINGER_MCP - WRIST||

If distance == 0:
    distance = 1.0  (fallback)

points_normalized = points / distance
```

This reference distance was chosen because it is relatively stable across different gestures (the palm structure doesn't change much).

## Step 2: Rotation normalization

A local coordinate system is built from three anchor points on the palm:

| Axis   | Definition                                        |
| ------ | ------------------------------------------------- |
| **Z**  | `RING_FINGER_MCP - WRIST` (along the palm)        |
| **X**  | Perpendicular to Z, in the direction of `RING_FINGER_MCP - INDEX_FINGER_MCP` (across the palm), orthogonalized via Gram-Schmidt |
| **Y**  | `cross(Z, X)` (normal to the palm plane)          |

A rotation matrix `R = [X | Y | Z]` is constructed from these three axes. Each landmark is then:

1. Translated so that the wrist is at the origin
2. Rotated by `R^T` to align to the canonical frame

``` text
for each landmark:
    v = landmark - wrist
    oriented = R^T @ v
```

After this step, all hand samples share the same orientation regardless of how the hand was held during capture.

## Step 3: Geometric feature extraction

In addition to the 63 normalized coordinates (21 x 3), **15 extra features** are computed from the scale-normalized (but not rotation-normalized) points:

### Finger curl angles (5 features)

The curl of each finger is measured as the angle at the PIP/IP joint:

| Feature               | Landmarks (a, b, c)              | Meaning                          |
| --------------------- | -------------------------------- | -------------------------------- |
| `THUMB_ANGLE`         | MCP(2), IP(3), TIP(4)            | Thumb curl angle at IP joint     |
| `INDEX_ANGLE`         | MCP(5), PIP(6), TIP(8)           | Index curl angle at PIP joint    |
| `MIDDLE_FINGER_ANGLE` | MCP(9), PIP(10), TIP(12)         | Middle finger curl               |
| `RING_FINGER_ANGLE`   | MCP(13), PIP(14), TIP(16)        | Ring finger curl                 |
| `PINKY_ANGLE`         | MCP(17), PIP(18), TIP(20)        | Pinky curl                       |

The angle is computed as:

``` text
angle(a, b, c) = arccos( dot(a-b, c-b) / (||a-b|| * ||c-b||) )
```

### Finger lengths (5 features)

The extended length of each finger (tip-to-base distance):

| Feature                  | Landmarks       | Meaning                          |
| ------------------------ | --------------- | -------------------------------- |
| `THUMB_LENGTH`           | TIP(4), MCP(2)  | Thumb extension                  |
| `INDEX_LENGTH`           | TIP(8), MCP(5)  | Index extension                  |
| `MIDDLE_FINGER_LENGTH`   | TIP(12), MCP(9) | Middle finger extension          |
| `RING_FINGER_LENGTH`     | TIP(16), MCP(13)| Ring finger extension            |
| `PINKY_LENGTH`           | TIP(20), MCP(17)| Pinky extension                  |

### Inter-finger features (5 features)

Spatial relationship between thumb and index finger:

| Feature                 | Computation                                                 |
| ----------------------- | ----------------------------------------------------------- |
| `THUMB_INDEX_ANGLE`     | Angle at THUMB_MCP(2) between THUMB_TIP(4) and INDEX_TIP(8) |
| `THUMB_INDEX_DISTANCE`  | Distance between INDEX_PIP(6) and THUMB_TIP(4)              |
| `THUMB_INDEX_X`         | `point[6].x - point[4].x` (x-axis separation)               |
| `THUMB_INDEX_Y`         | `point[7].y - point[5].y` (y-axis separation)               |
| `THUMB_INDEX_Z`         | `point[8].z - point[6].z` (z-axis separation)               |

## Final feature vector

The final feature vector has **78 dimensions**:

``` text
[x0, y0, z0, x1, y1, z1, ..., x20, y20, z20,   (63 values: rotated landmarks)
 THUMB_ANGLE, INDEX_ANGLE, ..., PINKY_ANGLE,      (5 values: curl angles)
 THUMB_LENGTH, INDEX_LENGTH, ..., PINKY_LENGTH,    (5 values: finger lengths)
 THUMB_INDEX_ANGLE, THUMB_INDEX_DISTANCE,          (5 values: inter-finger)
 THUMB_INDEX_X, THUMB_INDEX_Y, THUMB_INDEX_Z]
```

## MediaPipe landmark indices

For reference, here are the 21 MediaPipe hand landmarks:

``` text
 0: WRIST
 1: THUMB_CMC          5: INDEX_FINGER_MCP    9: MIDDLE_FINGER_MCP   13: RING_FINGER_MCP   17: PINKY_MCP
 2: THUMB_MCP          6: INDEX_FINGER_PIP   10: MIDDLE_FINGER_PIP   14: RING_FINGER_PIP   18: PINKY_PIP
 3: THUMB_IP           7: INDEX_FINGER_DIP   11: MIDDLE_FINGER_DIP   15: RING_FINGER_DIP   19: PINKY_DIP
 4: THUMB_TIP          8: INDEX_FINGER_TIP   12: MIDDLE_FINGER_TIP   16: RING_FINGER_TIP   20: PINKY_TIP
```
