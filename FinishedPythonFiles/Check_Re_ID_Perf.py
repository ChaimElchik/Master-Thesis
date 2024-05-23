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

from Mot_Cleaner import clean_file
from Detection_Video_Creator import TR_Vid_Creator
from Yolov8_Data_Prep import prepare_data_for_yolov8
from Deploy_Model import Deploy_Model
from Re_ID_Tracks import *
from ID_Matches import *
from ID_Match_DF_And_Dict import *
from Coordinates_3d_Df_Maker import *
from GT_To_TR_Mapping import *

# all matches when multiple matches in dataframe
def Check_Re_ID_Performance_v4(df, df_GT, gt_id_to_tr_id):
    df_perf = pd.DataFrame()
    Correctly_identified_within_margins = 0
    Not_identified_at_all = 0
    Identified_but_not_within_margins = 0
    too_many_fish = 0
    Not_a_real_fish = 0

    for frame in set(df_GT['frame']):
        if frame in set(df['frame']):
            for index, row in df_GT.loc[df_GT['frame'] == frame].iterrows():
                gt_id = row['id']
                if gt_id not in gt_id_to_tr_id:
                    Not_identified_at_all += 1
                else:
                    predicted_ids = gt_id_to_tr_id[gt_id]
                    matched = False
                    for id_match in predicted_ids:
                        match_row = df.loc[(df['frame'] == frame) & (df['id'] == id_match)]
                        if len(match_row) == 1:
                            gt_x = float(row['x'])
                            gt_y = float(row['y'])
                            gt_x_offset = float(row['x_offset'])
                            gt_y_offset = float(row['y_offset'])
                            pred_x = float(match_row['x'])
                            pred_y = float(match_row['y'])
                            pred_x_offset = float(match_row['x-offset'])
                            pred_y_offset = float(match_row['y-offset'])
                            dif_x = abs(abs(gt_x) - abs(pred_x))
                            dif_y = abs(abs(gt_y) - abs(pred_y))
                            tot_dif = dif_x + dif_y
                            if (dif_x < 11) & (dif_y < 11):
                                Correctly_identified_within_margins += 1
                                matched = True
                            elif (dif_x < 22) & (dif_y < 22):
                                Identified_but_not_within_margins += 1
                                matched = True
                            else:
                                Not_a_real_fish += 1
                            break  # Exit the loop once a match is found

                    if not matched:
                        Not_identified_at_all += 1

            if len(df['frame']) > len(df_GT['frame']):
                dif = len(df['frame']) - len(df_GT['frame'])
                too_many_fish += dif

    return [Correctly_identified_within_margins, Identified_but_not_within_margins, Not_a_real_fish, Not_identified_at_all, too_many_fish]
