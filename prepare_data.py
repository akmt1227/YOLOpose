import os
import cv2
import glob
import torch
import numpy as np
from ultralytics import YOLO

def extract_keypoints(video_path, yolo_model, seq_len=30):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    from collections import deque
    history = {} 
    sequences = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # track to keep IDs consistent
        results = yolo_model.track(frame, persist=True, verbose=False)
        
        if results and len(results[0].boxes) > 0:
            result = results[0]
            if result.keypoints is not None and result.boxes.id is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                
                # Normalize
                keypoints_norm = keypoints.copy()
                keypoints_norm[..., 0] /= width
                keypoints_norm[..., 1] /= height
                
                for i, track_id in enumerate(track_ids):
                    if track_id not in history:
                        history[track_id] = deque(maxlen=seq_len)
                        
                    kp_flat = keypoints_norm[i].flatten()
                    history[track_id].append(kp_flat)
                    
                    if len(history[track_id]) == seq_len:
                        # Append copy of current sequence
                        sequences.append(np.array(history[track_id]))
                        
    cap.release()
    return sequences

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_dir', type=str, default='dataset/normal', help='Path to normal videos')
    parser.add_argument('--abnormal_dir', type=str, default='dataset/abnormal', help='Path to abnormal videos')
    args = parser.parse_args()

    print("Loading YOLO model...")
    # 高速化のため L（Large）サイズから M（Medium）サイズに変更
    yolo_model = YOLO('yolov8m-pose.pt')
        
    classes = ["normal", "abnormal"]
    dataset_dirs = [args.normal_dir, args.abnormal_dir]
    
    X = []
    y = []
    
    for cls_idx, (cls_name, folder_path) in enumerate(zip(classes, dataset_dirs)):
        video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
        
        for video_file in video_files:
            print(f"Processing {video_file}...")
            seqs = extract_keypoints(video_file, yolo_model)
            print(f"Extracted {len(seqs)} sequences of 30 frames.")
            
            X.extend(seqs)
            y.extend([cls_idx] * len(seqs))
            
    if len(X) == 0:
        print("No data extracted. Please make sure .mp4 files are in dataset/normal and dataset/abnormal")
        return
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total sequences extracted: {len(X)}")
    print(f"Normal: {np.sum(y == 0)}, Abnormal: {np.sum(y == 1)}")
    
    np.save("X_data.npy", X)
    np.save("y_labels.npy", y)
    print("Saved extracted data to X_data.npy and y_labels.npy")

if __name__ == "__main__":
    main()
