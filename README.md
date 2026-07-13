# YOLOpose Anomaly Detection

This project uses YOLO-Pose (YOLO26l / YOLOv8l-pose) to detect human skeletons in MP4 videos, and processes the temporal sequence of keypoints using an LSTM to detect abnormal actions.

## Architecture & Design Choices

To avoid reinventing the wheel and ensure high readability, this project leverages several standard and powerful machine learning libraries as requested:

- **Ultralytics (YOLO)**: Used for state-of-the-art pose estimation and multi-person tracking.
- **PyTorch**: Used to build the core LSTM sequence model (`torch.nn.LSTM`).
- **timm (Hugging Face)**: Integrated into the LSTM model as an optional visual backbone to extract rich image features from the person's bounding box crops, complementing the YOLO keypoints.
- **sentence-transformers (Hugging Face)**: Used to encode text descriptions (e.g., "normal walking", "falling down") into a shared embedding space. This allows for zero-shot or few-shot anomaly detection by comparing video sequence embeddings to text descriptions.
- **pytorch-metric-learning**: Used in the training script to optimize the LSTM using contrastive loss (`NTXentLoss`), aligning the video sequence embeddings with the text descriptions.

## Files

- `models/anomaly_lstm.py`: The PyTorch LSTM model that fuses pose and visual features and projects them into the text embedding space.
- `train_metric_learning.py`: A script demonstrating how to train the LSTM using `pytorch-metric-learning` and contrastive loss.
- `main.py`: The main inference script that reads an MP4, runs YOLO-Pose, tracks individuals, passes their pose history to the LSTM, and renders the result.

## Usage

1. **Create Virtual Environment and Install dependencies using uv:**
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

2. **Run the inference pipeline:**
   Place an input video named `input.mp4` in the directory, or specify the path.
   ```bash
   uv run python main.py --input path/to/your_video.mp4 --output result.mp4
   ```

*(Note: The LSTM model is initialized with random weights in this template. To achieve accurate anomaly detection, you should collect training data (sequences of keypoints) and train the model using `train_metric_learning.py`)*
