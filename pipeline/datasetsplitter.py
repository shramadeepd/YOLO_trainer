import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch

class DatasetSplitter:
    def __init__(self, extracted_folder_path, split_ratios=(0.95, 0.025, 0.025), output_dir='datasets'):
        self.extracted_folder_path = extracted_folder_path
        self.output_dir = output_dir
        self.split_ratios = split_ratios
        
        # Ensure that split ratios add up to 1
        assert sum(self.split_ratios) == 1, "Split ratios must sum up to 1."
        
        # Check if the output directory exists, create it if not
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Directories for images and labels
        self.image_dir = os.path.join(self.extracted_folder_path, 'images')
        self.label_dir = os.path.join(self.extracted_folder_path, 'labels')
        
        # Check if directories exist
        if not os.path.exists(self.image_dir) or not os.path.exists(self.label_dir):
            raise FileNotFoundError("Image or label directory not found.")
        
        # Create output directories for train, valid, test sets
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)

    def get_image_label_files(self):
        """Get sorted lists of image and label files"""
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png','.jpeg'))])
        label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])

        # Ensure that the number of images matches the number of labels
        if len(image_files) != len(label_files):
            raise ValueError("Mismatch between the number of images and labels.")

        # Ensure images and labels correspond by filename
        assert all(os.path.splitext(img)[0] == os.path.splitext(lbl)[0] for img, lbl in zip(image_files, label_files)), \
            "Filenames of images and labels do not match."

        return image_files, label_files

    def split_data(self, image_files, label_files):
        """Split the dataset into train, validation, and test sets"""
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            image_files, label_files, test_size=(self.split_ratios[1] + self.split_ratios[2]), random_state=46
        )
        valid_images, test_images, valid_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=(self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2])), random_state=46
        )
        return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

    def move_files(self, file_list, source_dir, target_dir):
        """Move files to their respective directories"""
        for file_name in file_list:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))  # Using copy for testing

    def organize_data(self):
        """Organize and move the dataset to the output directory"""
        # Step 1: Get image and label files
        image_files, label_files = self.get_image_label_files()

        # Step 2: Split data
        train_images, train_labels, valid_images, valid_labels, test_images, test_labels = self.split_data(image_files, label_files)

        # Step 3: Move files to their respective directories
        self.move_files(train_images, self.image_dir, os.path.join(self.output_dir, 'train', 'images'))
        self.move_files(train_labels, self.label_dir, os.path.join(self.output_dir, 'train', 'labels'))
        self.move_files(valid_images, self.image_dir, os.path.join(self.output_dir, 'valid', 'images'))
        self.move_files(valid_labels, self.label_dir, os.path.join(self.output_dir, 'valid', 'labels'))
        self.move_files(test_images, self.image_dir, os.path.join(self.output_dir, 'test', 'images'))
        self.move_files(test_labels, self.label_dir, os.path.join(self.output_dir, 'test', 'labels'))

        # Step 4: Optionally clean up the extracted folder
        shutil.rmtree(self.extracted_folder_path, ignore_errors=True)
        print("Dataset split and organized successfully.")
