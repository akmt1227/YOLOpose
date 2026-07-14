# YOLOpose Anomaly Detection

Detect abnormal human actions in MP4 videos. YOLO-Pose extracts each person's
skeleton (17 keypoints) and tracks them across frames; an LSTM consumes the
temporal sequence of keypoints and is aligned—CLIP-style—to short text
descriptions ("normal walking" vs "falling down") so the decision can be made by
comparing the video embedding to those text anchors.

## Architecture & Design Choices

- **Ultralytics (YOLO)** — pose estimation and multi-person tracking. Default
  weights: `yolov8m-pose.pt` (configurable via `--model`; the repo also ships the
  larger `yolo26l-pose.pt`). Single source of truth: `utils/config.py:YOLO_MODEL`.
- **PyTorch (LSTM)** — the core sequence model (`models/anomaly_lstm.py`). Runs
  **pose-only by default**: each frame is a 51-dim feature = 17 keypoints ×
  `(x, y, confidence)`. A **timm** visual backbone is available as an opt-in
  (`PoseActionLSTM(use_visual=True)`) but is disabled by default to keep the model
  lean; enabling it requires wiring person crops into the pipeline.
- **sentence-transformers** — encodes the normal/abnormal text anchors
  (`all-MiniLM-L6-v2`, 384-dim). The LSTM projects video sequences into this same
  space.
- Training uses a **CLIP-style cross-entropy** against the two frozen text anchors,
  directly optimizing the same "normal vs abnormal" decision that inference makes.

### Feature engineering (important)

- **Person-centric normalization** (`utils/pose.py`): keypoints are centered on
  each person's bounding box and scaled by its size, making features translation-
  and scale-invariant (not tied to where the person is in the frame).
- **Confidence masking**: low-confidence / occluded joints are zeroed so YOLO's
  `(0, 0)` placeholders don't leak in, and the confidence itself is kept as a
  channel.
- `prepare_data.py` and `main.py` share the exact same feature code, so training
  and inference stay aligned.

## Files

- `models/anomaly_lstm.py` — the LSTM (pose-only by default; optional timm fusion).
- `utils/pose.py` — shared pose feature extraction/normalization + pipeline constants.
- `utils/config.py` — central config (YOLO weights, memory-pruning, smoothing).
- `prepare_data.py` — extracts keypoint sequences from videos → `X_data.npy`,
  `y_labels.npy`, `groups.npy` (source-video id per sequence, for leak-free splits).
- `train_metric_learning.py` — trains the LSTM, reports **validation accuracy** on
  a **video-level** split, and calibrates the decision margin → `lstm_model.pth`,
  `threshold.json`.
- `main.py` — inference: YOLO-Pose + tracking + LSTM, with majority-vote label
  smoothing and a calibrated threshold; renders the annotated MP4.
- `app.py` — Streamlit UI for the three stages below.

## Usage

1. **Install dependencies (uv):**
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

2. **Streamlit app (recommended):**
   ```bash
   uv run streamlit run app.py
   ```
   Then follow the three modes in order: **1. Data Prep → 2. Training → 3. Inference**.
   Put training videos in `dataset/normal/` and `dataset/abnormal/` first.

3. **Or run the pipeline from the CLI:**
   ```bash
   # 1) Extract keypoint sequences
   uv run python prepare_data.py --normal_dir dataset/normal --abnormal_dir dataset/abnormal

   # 2) Train + calibrate (prints Val Acc per epoch)
   uv run python train_metric_learning.py

   # 3) Inference on a new video
   uv run python main.py --input path/to/video.mp4 --output result.mp4
   ```

## Notes

- Feature dimension is **51** `(x, y, conf)`. If you have older `X_data.npy` or
  `lstm_model.pth` from a previous keypoint layout, regenerate them (re-run steps
  1–2); they are not compatible.
- Without training, `main.py` runs on random LSTM weights and its output is not
  meaningful—train first for real anomaly detection.
- Meaningful validation requires **at least 2 source videos** (ideally several per
  class); otherwise the split falls back to a random one with a warning.
