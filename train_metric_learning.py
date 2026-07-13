import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pytorch_metric_learning import losses
from sentence_transformers import SentenceTransformer
from models.anomaly_lstm import PoseActionLSTM
import numpy as np
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    if not os.path.exists("X_data.npy") or not os.path.exists("y_labels.npy"):
        print("Data files not found. Please run prepare_data.py first.")
        return
        
    # Load extracted data
    X = np.load("X_data.npy")
    y = np.load("y_labels.npy")
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    # small batch size since we have small data
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize our LSTM model
    model = PoseActionLSTM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Initialize Text Encoder for Zero-shot alignment (Hugging Face)
    print("Loading Text Encoder...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    loss_fn = losses.NTXentLoss(temperature=0.07)
    
    actions = [
        "person walking normally, standing, or doing routine activities", 
        "person falling down, slipping, fainting, fighting, or moving abnormally"
    ]
    
    # Pre-compute text embeddings
    with torch.no_grad():
        text_embs = text_encoder.encode(actions, convert_to_tensor=True)
        text_embs = torch.nn.functional.normalize(text_embs, p=2, dim=1) # Shape: (2, 384)
        
    num_epochs = 20
    model.train()
    
    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Get Video Embeddings from LSTM
            video_embs = model(batch_X)
            
            # Gather corresponding text embeddings for this batch
            batch_text_embs = text_embs[batch_y]
            
            # Contrastive Loss
            embeddings = torch.cat([video_embs, batch_text_embs], dim=0)
            
            # Pair labels: We assign unique IDs to each pair to align video i with text i
            batch_size = len(batch_X)
            pair_labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(device)
            
            loss = loss_fn(embeddings, pair_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
        
    # Save the trained model
    torch.save(model.state_dict(), "lstm_model.pth")
    print("Training complete! Model saved to lstm_model.pth")

if __name__ == "__main__":
    train()
