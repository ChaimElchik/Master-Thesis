import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import pandas as pd
from copy import deepcopy
from pathlib import Path
import matplotlib.pylab as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import os
from PIL import Image

def Deploy_Model(video_list,model):

    for video in video_list:
        video_path = 'vids/' + video
        cap = cv2.VideoCapture(video_path)


        # Get video properties for output
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define a video writer object
        output_path = 'output_tracking_' + video  # Specify the desired output file name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_dict = dict()  # For storing tracking data

        # Loop through the video frames
        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()

            if success:
            # Run YOLOv8 tracking on the frame
                results = model.track(frame, persist=True,tracker="bytetrack.yaml")
                sub_dict = {'IDS': results[0].boxes.id, 'Cor': results[0].boxes.xyxy}
                frame_dict[frame_num] = sub_dict

            # Draw tracking overlays on the frame
                annotated_frame = results[0].plot()

            # Write the annotated frame to the video writer
                writer.write(annotated_frame)

                frame_num += 1
            else:
                break

    # Release resources
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        rows = []
    # Iterate over the dictionary and append data to the list of rows
        for frame_number, value in frame_dict.items():
            ids = value['IDS']
            cor = value['Cor']
            if ids != None:
                for i in range(len(ids)):
    #             print(cor[i])
                    xmin = float(cor[i][0])
                    xmax = float(cor[i][2])
                    ymin = float(cor[i][1])
                    ymax = float(cor[i][3])
                    cx = float((xmin + xmax) / 2)
                    cy = float((ymin + ymax) / 2)
                    x_offset = cx - xmin  # Offset from the left edge
                    y_offset = cy - ymin  # Offset from the top edge
                    x = float(cor[i][0])
                    y = float(cor[i][1])
                    row_data = {'frame': frame_number, 'id': ids[i].item(), 'x': float(cor[i][0]), 'y': float(cor[i][1]), 'x-offset':x_offset,'y-offset':y_offset}
                    rows.append(row_data)

    # Create a DataFrame from the list of dictionaries
        dfv2 = pd.DataFrame(rows)

        res_name = 'df' + video.split('.')[0] + '_tr.csv'

        dfv2.to_csv(res_name, index=False)