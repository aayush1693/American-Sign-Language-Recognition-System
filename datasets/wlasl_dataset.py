import torch
from torch.utils.data import Dataset
import cv2
import os

class WLASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_dir, self.video_files[idx])
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
        return frames, self.video_files[idx]