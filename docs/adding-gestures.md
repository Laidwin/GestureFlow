# Adding New Gestures

This guide explains how to extend the system with custom gestures.

## 1. Choose a gesture name

Pick a clear, CamelCase name for your gesture (e.g. `PeaceSign`, `PointRight`, `SpiderMan`). This name will be used as the label throughout the pipeline.

## 2. Collect data

Open a terminal and run the capture command:

``` bash
uv run gesture-capture --gesture PeaceSign --name YourName
```

This opens the webcam. Position your hand and press **SPACE** to save each sample. Aim for:

- **At least 50-100 samples** per person per gesture (more is better)
- **Multiple contributors** to improve generalization across different hand shapes
- **Vary the hand position** slightly between captures (don't always hold it in the exact same spot)
- **Right hand only** (the current model is trained on right-hand data)

Press **ESC** when done. The data is saved to `data/raw/PeaceSign_YourName.json`.

Repeat for each contributor:

``` bash
uv run gesture-capture --gesture PeaceSign --name Alice
uv run gesture-capture --gesture PeaceSign --name Bob
```

## 3. Re-run the pipeline

### Normalize

``` bash
uv run gesture-normalize
```

This processes all files in `data/raw/`, including your new gesture.

### Fuse

Pass the full list of gestures (including the new one):

``` bash
uv run gesture-fuse --gestures CloseFist OkSign OpenHand ThumbsUp IndexUp PeaceSign
```

### Re-train

``` bash
uv run gesture-train
```

The training script will automatically pick up the new gesture from the fused data directory. Check the confusion matrix and classification report to verify that the new gesture is well-recognized.

## 4. Test

``` bash
uv run gesture-predict
```

The new gesture should now appear in the real-time prediction overlay.

## Tips

- **Confusing gestures:** If two gestures are often confused, collect more samples with exaggerated differences, or consider whether the gestures are truly distinguishable.
- **Removing a gesture:** Delete the corresponding files from `data/raw/`, `data/normalized/`, and `data/fused/`, then re-run fuse + train.
- **Updating the default gesture list:** Edit the `GESTURES` constant in `src/hand_gesture/fuse.py` to include your new gesture as a default.
- **Left hand support:** The current model is trained on right-hand data only. To support left hands, you would need to collect left-hand samples and potentially mirror the normalization.
