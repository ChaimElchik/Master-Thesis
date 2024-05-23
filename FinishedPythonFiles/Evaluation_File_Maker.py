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
from Check_Re_ID_Perf import *


def Create_Eval_File(file_list):
    with open("output_all_matches_for_dupes.txt", "w") as file:
            for file_name in file_list:

                parts = file_name.split('_')
                extracted_part = parts[0] + '_' + parts[1]

                path_name_df_re_id = 'df_' + extracted_part + '_Re_ID.csv'

                path_name_df_most_common = 'most_common_tr_gt_' + extracted_part + '.csv'

                path_before_re_id = 'df' + extracted_part + '_tr.csv'

                df_before_re_id = pd.read_csv(path_before_re_id)

                df_RE_ID = pd.read_csv(path_name_df_re_id)

                df_most_common = pd.read_csv(path_name_df_most_common)

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

                res = Check_Re_ID_Performance_v4(df_RE_ID,df_gt,gt_id_to_tr_id)

                file.write(str(file_name)+ "\n")
                file.write(str(len(set(df_gt['id']))) + " unique IDs GT\n")
                file.write(str(len(set(df_before_re_id['id']))) + " unique IDs before RE-ID\n")
                file.write(str(len(set(df_RE_ID['id']))) + " unique IDs after RE-ID\n")
                file.write(str((res[0]/len(df_gt)) * 100) + " % fish correctly identified within margin\n")
                file.write(str((res[1]/len(df_gt)) * 100) + " % fish correctly identified not within margin\n")
                file.write(str((res[3]/len(df_gt)) * 100) + " % fish not identified\n")
                file.write(str((res[2]/len(df_gt)) * 100) + " % not a real fish\n")
                file.write(str((res[0]/len(df_gt)) + (res[1]/len(df_gt)) + (res[3]/len(df_gt)) + (res[2]/len(df_gt))) + " Total\n")
                precision = res[0] / (res[0] + res[2])
                file.write("Precision: " + str(precision) + "\n")
                recall = res[0] / (res[0] + res[1] + res[3])
                file.write("Recall: " + str(recall) + "\n")
                f1 = (2 * precision * recall) / (precision + recall)
                file.write("F1: " + str(f1) + "\n")
                file.write("\n")  # Add a new line at the end