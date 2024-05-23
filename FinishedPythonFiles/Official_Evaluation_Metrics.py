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
from collections import defaultdict

def calculate_iou(gt_bbox, det_bbox):
    # Calculate Intersection over Union (IoU) between two bounding boxes
    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
    x1_det, y1_det, x2_det, y2_det = det_bbox

    xi1 = max(x1_gt, x1_det)
    yi1 = max(y1_gt, y1_det)
    xi2 = min(x2_gt, x2_det)
    yi2 = min(y2_gt, y2_det)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    det_area = (x2_det - x1_det) * (y2_det - y1_det)
    union_area = gt_area + det_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def map_detections_to_ground_truth(gt_df, det_df, id_map):
    # Remap detection IDs to match ground truth IDs using id_map
    det_df = det_df.copy()
    det_df['gt_id'] = -1

    for gt_id, det_ids in id_map.items():
        for det_id in det_ids:
            det_df.loc[det_df['id'] == det_id, 'gt_id'] = gt_id

    return det_df

def compute_hota(gt_df, det_df, id_map):
    # Remap detection IDs
    det_df = map_detections_to_ground_truth(gt_df, det_df, id_map)

    # Initialize HOTA score components
    overlap_scores = []
    association_scores = []

    # Iterate through each frame
    frames = gt_df['frame'].unique()
    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        det_frame = det_df[det_df['frame'] == frame]

        # Calculate IoU for each detection and corresponding ground truth
        for _, gt in gt_frame.iterrows():
            gt_bbox = [gt['x'], gt['y'], gt['x'] + gt['x_offset'], gt['y'] + gt['y_offset']]
            associated_detections = det_frame[det_frame['gt_id'] == gt['id']]
            max_iou = 0
            for _, det in associated_detections.iterrows():
                det_bbox = [det['x'], det['y'], det['x'] + det['x-offset'], det['y'] + det['y-offset']]
                iou = calculate_iou(gt_bbox, det_bbox)
                max_iou = max(max_iou, iou)

            if max_iou > 0:
                overlap_scores.append(max_iou)
                association_scores.append(1)
            else:
                association_scores.append(0)

    # Calculate final HOTA score
    if overlap_scores:
        avg_overlap = np.mean(overlap_scores)
    else:
        avg_overlap = 0.0

    if association_scores:
        avg_association = np.mean(association_scores)
    else:
        avg_association = 0.0

    hota_score = avg_overlap * avg_association
    return hota_score

def compute_mota(gt_df, det_df, id_map):
    # Remap detection IDs
    det_df = map_detections_to_ground_truth(gt_df, det_df, id_map)

    # Initialize MOTA score components
    num_misses = 0
    num_fps = 0
    num_switches = 0
    num_gt = len(gt_df)

    # Iterate through each frame
    frames = gt_df['frame'].unique()
    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        det_frame = det_df[det_df['frame'] == frame]

        # Track ground truth IDs
        gt_ids = set(gt_frame['id'])
        det_ids = set(det_frame['gt_id'])

        # Compute misses
        num_misses += len(gt_ids - det_ids)

        # Compute false positives
        num_fps += len(det_ids - gt_ids)

        # Compute identity switches
        for gt_id in gt_ids:
            det_matches = det_frame[det_frame['gt_id'] == gt_id]['id']
            if len(det_matches) > 1:
                num_switches += len(det_matches) - 1

    # Compute MOTA score
    mota_score = 1 - (num_misses + num_fps + num_switches) / num_gt
    return mota_score

def compute_assa(gt_df, det_df, id_map):
    # Remap detection IDs
    det_df = map_detections_to_ground_truth(gt_df, det_df, id_map)

    # Initialize AssA score components
    correct_associations = 0
    total_associations = 0

    # Iterate through each frame
    frames = gt_df['frame'].unique()
    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        det_frame = det_df[det_df['frame'] == frame]

        # Track ground truth IDs
        gt_ids = set(gt_frame['id'])
        det_ids = set(det_frame['gt_id'])

        # Count correct associations
        correct_associations += len(gt_ids.intersection(det_ids))

        # Count total associations
        total_associations += min(len(gt_ids), len(det_ids))

    # Compute AssA score
    if total_associations > 0:
        assa_score = correct_associations / total_associations
    else:
        assa_score = 0.0
    return assa_score

def compute_deta(gt_df, det_df, id_map, threshold=0.3):
    # Remap detection IDs
    det_df = map_detections_to_ground_truth(gt_df, det_df, id_map)

    # Initialize DetA score components
    correct_detections = 0
    total_detections = 0

    # Iterate through each frame
    frames = gt_df['frame'].unique()
    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        det_frame = det_df[det_df['frame'] == frame]

        # Iterate through each ground truth object
        for _, gt in gt_frame.iterrows():
            gt_id = gt['id']
            gt_bbox = [gt['x'], gt['y'], gt['x'] + gt['x_offset'], gt['y'] + gt['y_offset']]

            # Find the detection(s) associated with this ground truth object
            associated_detections = det_frame[det_frame['gt_id'] == gt_id]

            # Calculate the detection accuracy for each associated detection
            for _, det in associated_detections.iterrows():
                det_bbox = [det['x'], det['y'], det['x'] + det['x-offset'], det['y'] + det['y-offset']]
                iou = calculate_iou(gt_bbox, det_bbox)
                if iou >= threshold:
                    correct_detections += 1
                total_detections += 1

    # Compute DetA score
    if total_detections > 0:
        deta_score = correct_detections / total_detections
    else:
        deta_score = 0.0
    return deta_score

def compute_idf1(gt_df, det_df, id_map, threshold=0.3):
    # Remap detection IDs
    det_df = map_detections_to_ground_truth(gt_df, det_df, id_map)

    # Initialize IDF1 score components
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through each frame
    frames = gt_df['frame'].unique()
    for frame in frames:
        gt_frame = gt_df[gt_df['frame'] == frame]
        det_frame = det_df[det_df['frame'] == frame]

        # Track ground truth IDs
        gt_ids = set(gt_frame['id'])
        det_ids = set(det_frame['gt_id'])

        # Compute true positives
        for gt_id in gt_ids:
            associated_detections = det_frame[det_frame['gt_id'] == gt_id]
            if len(associated_detections) > 0:
                true_positives += 1

        # Compute false positives
        false_positives += len(det_ids - gt_ids)

        # Compute false negatives
        false_negatives += len(gt_ids - det_ids)

    # Compute precision, recall, and IDF1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    if precision + recall > 0:
        idf1_score = 2 * (precision * recall) / (precision + recall)
    else:
        idf1_score = 0.0
    return idf1_score

def Create_Eval_Metrics_File(file_list):

    with open("model_eval_all_matches_for_dupes.txt", "w") as file:
        for file_name in file_list:

            parts = file_name.split('_')
            extracted_part = parts[0] + '_' + parts[1]

            path_name_df_re_id = 'RE-IDS/df_' + extracted_part + '_Re_ID.csv'

            path_name_df_most_common = 'most_common/most_common_tr_gt_' + extracted_part + '.csv'

            df_RE_ID = pd.read_csv(path_name_df_re_id)

            df_most_common = pd.read_csv(path_name_df_most_common)

            from collections import defaultdict

            # Create a defaultdict to store multiple values for each key
            gt_id_to_tr_id = defaultdict(list)

            # Iterate through each row of the DataFrame
            for index, row in df_most_common.iterrows():
                # Add the tr_id to the list of values for the corresponding gt_id
                gt_id_to_tr_id[row['gt_id']].append(row['tr_id'])

            # Convert defaultdict to a regular dictionary
            gt_id_to_tr_id = dict(gt_id_to_tr_id)

            path_gt = 'mots/' + extracted_part + '_clean.txt'

            df_gt = pd.read_csv(path_gt)

            gt_df = df_gt
            id_map = gt_id_to_tr_id
            det_df = df_RE_ID

            print(file_name)
            file.write(str(file_name)+ "\n")
            HOTA = compute_hota(gt_df, det_df, id_map)
            file.write("HOTA Score: " + str(HOTA) + "\n")
            MOTA = compute_mota(gt_df, det_df, id_map)
            file.write("MOTA Score: " + str(MOTA) + "\n")
            AssA = compute_assa(gt_df, det_df, id_map)
            file.write("AssA Score: " + str(AssA) + "\n")
            DetA = compute_deta(gt_df, det_df, id_map)
            file.write("DetA Score: " + str(DetA) + "\n")
            IDF1 = compute_idf1(gt_df, det_df, id_map)
            file.write("IDF1 Score: " + str(IDF1) + "\n")
            file.write("\n")  # Add a new line at the end


def parse_block(block):
    lines = block.strip().split('\n')
    if len(lines) < 6:  # Check if there are enough lines to parse
        return None
    data = {}
    data['ID'] = lines[0].strip()
    data['HOTA Score'] = float(lines[1].split(':')[1])
    data['MOTA Score'] = float(lines[2].split(':')[1])
    data['AssA Score'] = float(lines[3].split(':')[1])
    data['DetA Score'] = float(lines[4].split(':')[1])
    data['IDF1 Score'] = float(lines[5].split(':')[1])

    return data

# Define a function to parse the entire text file
def parse_txt_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    blocks = text.strip().split('\n\n')  # Split text into blocks
    data_list = [parse_block(block) for block in blocks if parse_block(block) is not None]
#     print(data_list)
    return data_list

def Create_Table_Metrics():
    # Path to the text file
    file_path = 'model_eval_all_matches_for_dupes.txt'

    # Parse the text file
    data_list = parse_txt_file(file_path)

    # Convert the parsed data into a DataFrame
    df = pd.DataFrame(data_list)

    output_csv_path = 'model_eval_table_all_matches_for_dupes.csv'
    df.to_csv(output_csv_path, index=False)

