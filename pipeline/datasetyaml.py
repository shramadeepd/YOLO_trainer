import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch


class DatasetYamlWriter:
    def __init__(self,num_classes,class_names,output_file="dataset_path.yaml"):
        # Fixed paths for train, validation, and test datasets
        self.train_path = "datasets/train/images"
        self.val_path = "datasets/valid/images"
        self.test_path = "datasets/test/images"
        
        # Default class names and number of classes
        self.num_classes = num_classes
        # self.class_names = [
        #     "bicycles", "buses", "chimneys", "crosswalks", "fire hydrants", "motorcycles",
        #     "parking meters", "stairs", "taxis", "tractors", "traffic lights", "vehicles"
        # ]
        self.class_names = class_names
        
        self.output_file = output_file

    def generate_yaml_content(self):
        """Create the YAML content"""
        yaml_content = f"""
train: {self.train_path}
val: {self.val_path}
test: {self.test_path}

nc: {self.num_classes} # Number of classes
names: {self.class_names}
"""
        return yaml_content

    def write_yaml(self):
        """Write the YAML content to the output file"""
        yaml_content = self.generate_yaml_content()
        
        # Write the YAML content to the output file
        with open(self.output_file, "w") as file:
            file.write(yaml_content)
        print(f"YAML file '{self.output_file}' written successfully.")
