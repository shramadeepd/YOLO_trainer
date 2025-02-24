import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
from .datasetsplitter import DatasetSplitter
from .datasetyaml import DatasetYamlWriter
from .trainer import YOLOTrainer
from .utils import download_and_unzip

class FullPipeline:
    def __init__(self,model="yolo11n.pt",epochs=500,batch_size=8,LS_ID=None):
        self.extracted_folder_path = 'data'
        self.data_config = "dataset_path.yaml"
        self.output_dir = "datasets"
        # self.num_classes=num_classes
        # self.class_names=class_names
        self.model = model
        self.epochs=epochs
        self.batch_size=batch_size
        self.LS_ID = LS_ID
    
        
        # if num_classes or class_names is None:
        #     print('fill the mandatory parameters')

    def run(self):
        # if self.LS_ID is not None:
        #     download_and_unzip(self.LS_ID)


        # # Step 2: Write the YAML configuration
        # yaml_writer = DatasetYamlWriter()
        # yaml_writer.write_yaml()

        # Check if datasets folder already exists
        if os.path.exists(self.output_dir):
            print(f"'{self.output_dir}' already exists. Skipping dataset split.")
            if not os.path.exists(self.data_config):
                print("Please create the config file")
            else:
                print("noted??")
            # yaml_writer = DatasetYamlWriter()
            # yaml_writer.write_yaml()
        else:
            # Split dataset
            if self.LS_ID is not None:
                download_and_unzip(self.LS_ID)

                yaml_writer = DatasetYamlWriter()
                yaml_writer.write_yaml()
                splitter = DatasetSplitter(self.extracted_folder_path, output_dir=self.output_dir)
                splitter.organize_data()

        # Train the model
        trainer = YOLOTrainer(self.model,data_config=self.data_config,epochs=self.epochs,batch_size=self.batch_size)
        trainer.start_training()

