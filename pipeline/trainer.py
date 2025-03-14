import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch


class YOLOTrainer:
    def __init__(self, model,epochs, data_config="dataset_path.yaml", batch_size=8):
        self.model_path = model
        self.data_config = data_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        """Load the YOLO model"""
        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        print(f"Model loaded successfully on {self.device}.")

    def train_model(self):
        """Train the YOLO model"""
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model first using `load_model()`.")
        
        print(f"Training model for {self.epochs} epochs with batch size {self.batch_size}...")

        try:
            print(self.data_config)
            results = self.model.train(data=self.data_config, epochs=self.epochs, batch=self.batch_size)
            print("Training completed successfully.")
            return results
        except Exception as e:
            print(f"An error occurred during training: {e}")
            return None

    def start_training(self):
        """Start the full process of loading and training the model"""
        self.load_model()
        return self.train_model()
