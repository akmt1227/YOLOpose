import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import warnings
from models.anomaly_lstm import PoseActionLSTM

warnings.filterwarnings('ignore')

def process_video(input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load YOLO Pose Model
    print("Loading YOLOv8m-pose.pt...")
    yolo_model = YOLO('yolov8m-pose.pt')
        
    yolo_model.to(device)

    # 2. Load our LSTM Sequence Model
    print("Initializing LSTM Anomaly Model...")
    lstm_model = PoseActionLSTM().to(device)
    import os
    if os.path.exists("lstm_model.pth"):
        print("Loading trained weights from lstm_model.pth...")
        lstm_model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
    else:
        print("Warning: lstm_model.pth not found. Using random weights!")
    lstm_model.eval() # Set to inference mode
    
    # 3. Load Text Encoder (Sentence-Transformers)
    print("Loading Sentence-Transformers for Zero-Shot Classification...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    # Encode anchor texts representing Normal and Abnormal classes
    normal_text = "person walking normally, standing, or doing routine activities"
    abnormal_text = "person falling down, slipping, fainting, fighting, or moving abnormally"
    with torch.no_grad():
        anchor_embs = text_encoder.encode([normal_text, abnormal_text], convert_to_tensor=True)
        anchor_embs = torch.nn.functional.normalize(anchor_embs, p=2, dim=1) # Shape: (2, 384)

    # 4. Video IO setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seq_len = 30
    # Store history for each tracked person (id -> deque of keypoints)
    history = {} 

    print(f"Starting video processing. Input: {input_path}, Output: {output_path}")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO with tracking enabled to associate skeletons across frames
        results = yolo_model.track(frame, persist=True, verbose=False)
        
        annotated_frame = frame.copy()
        
        if results and len(results[0].boxes) > 0:
            result = results[0]
            # Automatically plot YOLO bounding boxes and skeletons
            annotated_frame = result.plot() 
            
            # Extract keypoints
            if result.keypoints is not None and result.boxes.id is not None:
                keypoints = result.keypoints.xy.cpu().numpy() # (num_people, 17, 2)
                track_ids = result.boxes.id.int().cpu().numpy()
                
                # Normalize keypoints by image dimensions (scale-invariant)
                keypoints_norm = keypoints.copy()
                keypoints_norm[..., 0] /= width
                keypoints_norm[..., 1] /= height
                
                for i, track_id in enumerate(track_ids):
                    if track_id not in history:
                        history[track_id] = deque(maxlen=seq_len)
                    
                    # Flatten the 17x2 keypoints to a 34-dim vector
                    kp_flat = keypoints_norm[i].flatten()
                    history[track_id].append(kp_flat)
                    
                    # If we have accumulated enough frames for this person, run LSTM
                    if len(history[track_id]) == seq_len:
                        seq_tensor = torch.tensor(np.array(history[track_id]), dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            # Forward pass through LSTM
                            video_emb = lstm_model(seq_tensor, crops=None)
                            
                            # Zero-shot inference: Calculate Cosine Similarity with text anchors
                            # video_emb: (1, 384), anchor_embs: (2, 384)
                            sims = torch.mm(video_emb, anchor_embs.t())[0]
                            
                            # Note: Model is untrained, so outputs are essentially random at this stage.
                            # In a real scenario, the LSTM would be pre-trained using train_metric_learning.py
                            is_abnormal = sims[1] > sims[0] + 0.05
                            
                        # Overlay Anomaly Status
                        bbox = result.boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        label = "Abnormal!" if is_abnormal else "Normal"
                        color = (0, 0, 255) if is_abnormal else (0, 255, 0)
                        
                        # Draw label background and text
                        cv2.rectangle(annotated_frame, (x1, max(0, y1 - 40)), (x1 + 140, max(0, y1)), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 5, max(0, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                    
        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"Finished processing! Video saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Pose + LSTM Anomaly Detection")
    parser.add_argument('--input', type=str, default='input.mp4', help='Path to input MP4 video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output MP4 video')
    args = parser.parse_args()
    
    process_video(args.input, args.output)
