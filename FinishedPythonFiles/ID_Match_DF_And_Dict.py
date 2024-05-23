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

def Match_DF_Maker(extracted_part):

    path_name = 'df_pre_Re_ID_' + extracted_part + '.csv'

    df_pre_RE_ID = pd.read_csv(path_name)

    # Group by 'left_id' and count occurrences of each 'right_id'
    most_common_tr_gt = df_pre_RE_ID.groupby('tr_id')['gt_id'].agg(lambda x: x.value_counts().index[0])

    # Convert Series to DataFrame
    most_common_tr_gt = most_common_tr_gt.reset_index(name='gt_id')

    end_name = 'most_common_' + extracted_part + '.csv'

    most_common_tr_gt.to_csv(end_name, index=False)

def Match_Dict_Maker(path_name_df_most_common):
    df_most_common = pd.read_csv(path_name_df_most_common)

    # Create a defaultdict to store multiple values for each key
    left_id_to_right_id = defaultdict(list)

    # Iterate through each row of the DataFrame
    for index, row in df_most_common.iterrows():
        # Add the tr_id to the list of values for the corresponding gt_id
        left_id_to_right_id[row['gt_id']].append(row['tr_id'])

    # Convert defaultdict to a regular dictionary
    left_id_to_right_id = dict(left_id_to_right_id)

    return left_id_to_right_id