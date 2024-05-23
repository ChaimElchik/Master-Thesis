import cv2
import pandas as pd
from copy import deepcopy
from pathlib import Path
import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import cv2
from scipy.io import loadmat
from copy import deepcopy
from PIL import Image

import cv2
import pandas as pd
import os
import argparse

def prepare_data_for_yolov8(video_paths, csv_paths, output_dir):
    """
    Prepares data from multiple video files and CSVs for YOLOv8 training or inference.

    Args:
        video_paths (list): List of paths to the video files.
        csv_paths (list): List of paths to the CSV files containing object coordinates.
        output_dir (str): The output directory where images and labels will be saved.
    """

    # Input validation
    if len(video_paths) != len(csv_paths):
        raise ValueError("The number of video paths and CSV paths must match.")

    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Process each video and corresponding CSV
    for video_path, csv_path in zip(video_paths, csv_paths):
        base_filename = os.path.splitext(os.path.basename(video_path))[0]

        # Load the CSV data
        data = pd.read_csv(csv_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save the image
            image_path = os.path.join(images_dir, f'{base_filename}_{frame_index}.jpg')
            cv2.imwrite(image_path, frame)

            # Create the label file
            label_path = os.path.join(labels_dir, f'{base_filename}_{frame_index}.txt')
            frame_data = data[data['frame'] == frame_index]

            with open(label_path, 'w') as f:
                for _, row in frame_data.iterrows():
                    obj_id = row['id']
                    x = row['x']
                    y = row['y']
                    x_offset = row['x_offset']
                    y_offset = row['y_offset']
                    image_width = frame.shape[1]
                    image_height = frame.shape[0]

                    # Calculate bounding box coordinates
                    top_left_x = x - x_offset
                    top_left_y = y - y_offset
                    width = 2 * x_offset
                    height = 2 * y_offset

                    # Normalize coordinates
                    center_x = (top_left_x + width / 2) / image_width
                    center_y = (top_left_y + height / 2) / image_height
                    width = width / image_width
                    height = height / image_width

                    # Assume class label is always 0 (you may need to change this)
                    class_label = 0

                    f.write(f"{class_label} {center_x} {center_y} {width} {height}\n")

            frame_index += 1

        cap.release()

# Example usage
video_paths = ['../vids/349_2.mp4','../vids/406_1.mp4', '../vids/161_2.mp4']
csv_paths = ['349_2_clean.txt','406_1_clean.txt','161_2_clean.txt']
output_dir = 'dataset_V2_Val'
prepare_data_for_yolov8(video_paths, csv_paths, output_dir)


# val  ['349_2_clean.txt','406_1_clean.txt','161_2_clean.txt']
