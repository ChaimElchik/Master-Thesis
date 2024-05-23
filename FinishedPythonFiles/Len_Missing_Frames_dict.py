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
import json

from Mot_Cleaner import *
from Detection_Video_Creator import *
from Yolov8_Data_Prep import *
from Deploy_Model import *
from Re_ID_Tracks import *
from ID_Matches import *
from ID_Match_DF_And_Dict import *
from Coordinates_3d_Df_Maker import *
from GT_To_TR_Mapping import *
from Check_Re_ID_Perf import *
from Evaluation_File_Maker import *

# Custom JSON encoder to handle sets
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

# Check if list is nested
def is_nested(lst):
    """
    Check if the list contains any nested lists.
    """
    return any(isinstance(i, list) for i in lst)

# Flatten list
def flatten(lst):
    """
    Flatten a nested list.
    """
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list

# Create missing frames dictionary
def Create_Len_Missing_Frames_Dict(file_list):
    end_dict = dict()
    for file_name in file_list:
        parts = file_name.split('_')
        extracted_part = parts[0] + '_' + parts[1]
        path_gt = 'mots/' + extracted_part + '_clean.txt'
        path_tr = 'RE-IDS/df_' + extracted_part + '_Re_ID.csv'
        df_gt = pd.read_csv(path_gt)
        df_tr = pd.read_csv(path_tr)
        d = dict()
        df_ = pd.DataFrame()

        path_name_df_most_common = 'most_common/most_common_tr_gt_' + extracted_part + '.csv'

        df_most_common = pd.read_csv(path_name_df_most_common)

        # Create a defaultdict to store multiple values for each key
        gt_id_to_tr_id = defaultdict(list)

        # Iterate through each row of the DataFrame
        for index, row in df_most_common.iterrows():
            # Add the tr_id to the list of values for the corresponding gt_id
            gt_id_to_tr_id[row['gt_id']].append(row['tr_id'])

        # Convert defaultdict to a regular dictionary
        gt_id_to_tr_id = dict(gt_id_to_tr_id)
        for id_ in set(df_gt['id']):
            df_gt_temp = df_gt.loc[df_gt['id'] == id_]

            if id_ in gt_id_to_tr_id:
                predicted_ids = gt_id_to_tr_id[id_]

                matched = False
                frames_list = []
                for id_match in predicted_ids:
                    match_row = df_tr.loc[df_tr['id'] == id_match]
                    frames_list.append(list(match_row['frame']))
                if is_nested(frames_list):
                    frames_list = flatten(frames_list)
                d[id_] = set(df_gt_temp['frame']) - set(frames_list)
            else:
                d[id_] = "Missing match"
        end_dict[extracted_part] = d

    end_dict_lens = end_dict
    # Iterate over the nested dictionary
    for key, value in end_dict_lens.items():
        for inner_key, inner_value in value.items():
            # Replace the set with its length
            if inner_value != "Missing match":
                end_dict_lens[key][inner_key] = len(inner_value)
            else:
                end_dict_lens[key][inner_key] = "Missing match"

    # Specify the filename
    filename = 'dictionary_missing_frames_len.json'
    # Write dictionary to JSON file using the custom encoder
    try:
        with open(filename, 'w') as json_file:
            json.dump(end_dict_lens, json_file, indent=4, cls=CustomJSONEncoder)
        print(f'Dictionary has been written to {filename}')
    except IOError as e:
        print(f'An error occurred while writing to the file: {e}')