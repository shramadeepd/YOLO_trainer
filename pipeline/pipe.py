import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
from .datasetsplitter import DatasetSplitter
from .datasetyaml import DatasetYamlWriter
from .trainer import YOLOTrainer

class FullPipeline:
    def __init__(self,num_classes,class_names,model="yolo11n.pt",epochs=500,batch_size=8):
        self.extracted_folder_path = 'data'
        self.data_config = "dataset_path.yaml"
        self.output_dir = "datasets"
        self.num_classes=num_classes
        self.class_names=class_names
        self.model = model
        self.epochs=epochs
        self.batch_size=batch_size
        
        if num_classes or class_names is None:
            print('fill the mandatory parameters')

    def run(self):
        # Step 1: Check if datasets folder already exists
        if os.path.exists(self.output_dir):
            print(f"'{self.output_dir}' already exists. Skipping dataset split.")
        else:
            # Step 1: Split dataset
            splitter = DatasetSplitter(self.extracted_folder_path, output_dir=self.output_dir)
            splitter.organize_data()

        # Step 2: Write the YAML configuration
        yaml_writer = DatasetYamlWriter(self.num_classes,self.class_names)
        yaml_writer.write_yaml()

        # Step 3: Train the model
        trainer = YOLOTrainer(self.model,data_config=self.data_config,epochs=self.epochs,batch_size=self.batch_size)
        trainer.start_training()