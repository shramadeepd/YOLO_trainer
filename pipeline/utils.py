import random
import os
import shutil
import requests
from dotenv import load_dotenv
import os
import traceback
import requests, zipfile, os, shutil,yaml, json, socket, time, pathlib 
from logger import logger
from io import BytesIO
load_dotenv()

def split_dataset(image_files, label_files, split_ratios, seed=46):
    """Custom function to split the dataset manually without using sklearn"""
    assert sum(split_ratios) == 1, "Split ratios must sum up to 1."

    # Ensure reproducibility
    random.seed(seed)

    # Shuffle data
    combined = list(zip(image_files, label_files))
    random.shuffle(combined)
    image_files, label_files = zip(*combined)

    # Compute split indices
    total = len(image_files)
    train_end = int(split_ratios[0] * total)
    valid_end = train_end + int(split_ratios[1] * total)

    # Split into train, validation, and test sets
    train_images, train_labels = image_files[:train_end], label_files[:train_end]
    valid_images, valid_labels = image_files[train_end:valid_end], label_files[train_end:valid_end]
    test_images, test_labels = image_files[valid_end:], label_files[valid_end:]

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def move_files(file_list, source_dir, target_dir):
    """Move files to their respective directories"""
    os.makedirs(target_dir, exist_ok=True)
    for file_name in file_list:
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))  # Using copy for testing

def download_and_unzip(LS_ID):
    try:
        # current_time = int(time.time())
        # update agent status as onprocess
        # AgentYoloTrainer.update_agent_status(self, self.host, status='onprocess')
        # ls_id= 4
        unzipped_folder_name = 'data'
        url = f"http://140.238.247.130:8080/api/projects/{LS_ID}/export?exportType=YOLO"
        token=os.getenv("LS_TOKEN")
        # logger.info(url)
        headers = {"Authorization": f"Token  {token}"}
        print('Data download started.......')
        response = requests.get(url, headers=headers)
        print(response.status_code)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
                os.makedirs(unzipped_folder_name, exist_ok=True)
                zip_ref.extractall(unzipped_folder_name)
            logger.info('Download ✅')
            print('Download ✅')
        else:
            logger.error("Downloading failed from label studio", exc_info=1)
            print("error in downloading data")


    except Exception as e:
        # traceback.print_exc()
        logger.error("Download and unzip failed ", exc_info=1)