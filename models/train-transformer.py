import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os
import json

class WLASLDataset(Dataset):
    def __init__(self, video_dir, annotations_file, transform=None):
        self.video_dir = video_dir
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.annotations[idx]['video'])
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        frames = torch.stack(frames)
        label = self.annotations[idx]['label']
        return frames, label

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

def train_model(video_dir, annotations_file, vocab_size, num_epochs=10, batch_size=4, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = WLASLDataset(video_dir, annotations_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for frames, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(frames, labels)
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'models/transformer_model.pth')

if __name__ == '__main__':
    video_dir = 'path/to/WLASL/videos'
    annotations_file = 'path/to/WLASL/annotations.json'
    vocab_size = 30  # Adjust based on your character set
    train_model(video_dir, annotations_file, vocab_size)