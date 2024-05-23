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

def pre_maker(df_tr,df_gt):
    df_pre = pd.DataFrame()
    for frame in set(df_tr['frame']):
        if frame in set(df_gt['frame']):
            for index, row in df_tr.loc[df_tr['frame'] == frame].iterrows():
                tr_id = row['id']
                tuple_tr = (row['x'],row['y'])
                match_dict = dict()
                for index2, row2 in df_gt.loc[df_gt['frame'] == frame].iterrows():
                    gt_id = row2['id']
                    tuple_gt = (row2['x'],row2['y'])
                    tuple_dif = abs(tuple_differences(tuple_tr, tuple_gt)[0]) + abs(tuple_differences(tuple_tr, tuple_gt)[1])
                    match_dict[(tr_id,gt_id)] = tuple_dif
                min_key = min(match_dict, key=match_dict.get)
                print(min_key[0],min_key[1],frame)

                row_dict = {'tr_id':min_key[0] ,'gt_id':min_key[1], 'difference':match_dict[min_key], 'frame':frame}

                df_row = pd.DataFrame.from_dict([row_dict])

                df_pre = pd.concat([df_pre,df_row],ignore_index=True)
    return df_pre