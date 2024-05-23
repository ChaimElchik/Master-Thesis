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
from Missing_Frames_dict import *
from Len_Missing_Frames_dict import *
from Official_Evaluation_Metrics import *

def parse_block(block):
    lines = block.strip().split('\n')
    if len(lines) < 12:  # Check if there are enough lines to parse
        return None
    data = {}
    data['ID'] = lines[0].strip()
    data['GT_unique_IDs'] = int(lines[1].split()[0])
    data['Before_RE_ID_unique_IDs'] = int(lines[2].split()[0])
    data['After_RE_ID_unique_IDs'] = int(lines[3].split()[0])
    data['Fish_correctly_identified_within_margin'] = float(lines[4].split()[0]) / 100
    data['Fish_correctly_identified_not_within_margin'] = float(lines[5].split()[0]) / 100
    data['Fish_not_identified'] = float(lines[6].split()[0]) / 100
    data['Not_a_real_fish'] = float(lines[7].split()[0]) / 100
#     data['Total'] = float(lines[8].split()[0])
    data['Precision'] = float(lines[9].split(':')[1])
    data['Recall'] = float(lines[10].split(':')[1])
    data['F1'] = float(lines[11].split(':')[1])

    return data

# Define a function to parse the entire text file
def parse_txt_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    blocks = text.strip().split('\n\n')  # Split text into blocks
    data_list = [parse_block(block) for block in blocks if parse_block(block) is not None]
    return data_list



if __name__ == "__main__":
    # video files list
    video_list = [
        '8_1.mp4', '8_2.mp4', '10_1.mp4', '10_2.mp4', '13_1.mp4', '13_2.mp4', '14_1.mp4', '14_2.mp4',
        '15_1.mp4', '15_2.mp4', '16_1.mp4', '16_2.mp4', '20_1.mp4', '20_2.mp4', '23_1.mp4', '23_2.mp4',
        '24_1.mp4', '24_2.mp4', '26_1.mp4', '26_2.mp4', '29_1.mp4', '29_2.mp4', '32_1.mp4', '32_2.mp4',
        '33_1.mp4', '33_2.mp4', '41_1.mp4', '41_2.mp4', '42_1.mp4', '42_2.mp4', '43_1.mp4', '43_2.mp4',
        '46_1.mp4', '46_2.mp4', '48_1.mp4', '48_2.mp4', '51_1.mp4', '51_2.mp4', '54_1.mp4', '54_2.mp4',
        '58_1.mp4', '58_2.mp4', '60_1.mp4', '60_2.mp4', '61_1.mp4', '61_2.mp4', '62_1.mp4', '62_2.mp4',
        '70_1.mp4', '70_2.mp4', '83_1.mp4', '83_2.mp4', '92_1.mp4', '92_2.mp4', '116_1.mp4', '116_2.mp4',
        '122_1.mp4', '122_2.mp4', '129_1.mp4','129_2.mp4','134_1.mp4', '134_2.mp4', '143_1.mp4', '143_2.mp4','161_1.mp4','161_2.mp4', '168_1.mp4', '168_2.mp4',
        '175_1.mp4', '175_2.mp4','183_1.mp4','183_2.mp4', '186_1.mp4', '186_2.mp4', '187_1.mp4', '187_2.mp4', '189_1.mp4', '189_2.mp4',
        '191_1.mp4', '191_2.mp4', '194_1.mp4', '194_2.mp4', '195_1.mp4', '195_2.mp4', '198_1.mp4', '198_2.mp4',
        '202_1.mp4', '202_2.mp4', '204_1.mp4', '204_2.mp4', '206_1.mp4', '206_2.mp4', '212_1.mp4', '212_2.mp4',
        '214_1.mp4', '214_2.mp4', '217_1.mp4', '217_2.mp4', '223_1.mp4', '223_2.mp4','231_1.mp4','231_2.mp4', '234_1.mp4', '234_2.mp4',
        '236_1.mp4', '236_2.mp4', '239_1.mp4', '239_2.mp4', '240_1.mp4', '240_2.mp4', '242_1.mp4', '242_2.mp4',
        '244_1.mp4', '244_2.mp4', '247_1.mp4', '247_2.mp4', '249_1.mp4', '249_2.mp4', '260_1.mp4', '260_2.mp4',
        '261_1.mp4', '261_2.mp4', '265_1.mp4', '265_2.mp4', '268_1.mp4', '268_2.mp4', '270_1.mp4', '270_2.mp4',
        '278_1.mp4', '278_2.mp4', '280_1.mp4', '280_2.mp4', '283_1.mp4', '283_2.mp4', '289_1.mp4', '289_2.mp4',
        '294_1.mp4', '294_2.mp4', '295_1.mp4', '295_2.mp4', '296_1.mp4', '296_2.mp4', '297_1.mp4', '297_2.mp4',
        '301_1.mp4', '301_2.mp4', '302_1.mp4', '302_2.mp4', '303_1.mp4', '303_2.mp4', '304_1.mp4', '304_2.mp4',
        '308_1.mp4', '308_2.mp4', '309_1.mp4', '309_2.mp4', '312_1.mp4', '312_2.mp4', '349_1.mp4', '349_2.mp4',
        '368_1.mp4', '368_2.mp4', '371_1.mp4', '371_2.mp4', '373_1.mp4', '373_2.mp4', '374_1.mp4', '374_2.mp4',
        '376_1.mp4', '376_2.mp4', '378_1.mp4', '378_2.mp4', '384_1.mp4', '384_2.mp4', '392_1.mp4', '392_2.mp4', '406_1.mp4','406_2.mp4',
        '432_1.mp4', '432_2.mp4', '437_1.mp4', '437_2.mp4'
    ]

    file_list = [filename.replace('.mp4', '_tr.csv') for filename in video_list]

    for file_name in file_list:

        parts = file_name.split('_')
        extracted_part = parts[0] + '_' + parts[1]
        tr_path = 'df_' + extracted_part  + '_Re_ID.csv'

        gt_path = 'mots/' + extracted_part + '_clean.txt'

        df_tr = pd.read_csv(tr_path)

        df_gt = pd.read_csv(gt_path)

        df_pre = pre_maker(df_tr,df_gt)

        end_name = 'df_pre_Re_ID_' + extracted_part + '.csv'

        df_pre.to_csv(end_name, index=False)

    for file_name in file_list:

        parts = file_name.split('_')
        extracted_part = parts[0] + '_' + parts[1]

        path_name = 'df_pre_Re_ID_' + extracted_part + '.csv'

        df_pre_RE_ID = pd.read_csv(path_name)

        # Group by 'left_id' and count occurrences of each 'right_id'
        most_common_tr_gt = df_pre_RE_ID.groupby('tr_id')['gt_id'].agg(lambda x: x.value_counts().index[0])

        # Convert Series to DataFrame
        most_common_tr_gt = most_common_tr_gt.reset_index(name='gt_id')

        end_name = 'most_common_tr_gt_' + extracted_part + '.csv'

        most_common_tr_gt.to_csv(end_name, index=False)

    # Create Evaluation File with own metrics, precision, recall and f1. Output file named 'output_all_matches_for_dupes.txt'
    Create_Eval_File(file_list)

    # Path to the text file
    file_path = 'output_all_matches_for_dupes.txt'

    # Parse the text file
    data_list = parse_txt_file(file_path)

    # Convert the parsed data into a DataFrame
    df = pd.DataFrame(data_list)

    # write to csv file
    output_csv_path = 'output_table_all_matches_for_dupes.csv'
    df.to_csv(output_csv_path, index=False)

    # Create a dictionary with all missing frames per id per video when compared to ground truth. (GT IDs used) dictionary_missing_frames.json
    Create_Missing_Frames_Dict(file_list)

    # Create a dictionary with amount of missing frames per id per video when compared to ground truth. (GT IDs used) dictionary_missing_frames_len.json
    Create_Len_Missing_Frames_Dict(file_list)

    # Create text file with HOTA, MOTA, AssA, DetA and IDF1 scores "model_eval_all_matches_for_dupes.txt"
    Create_Eval_Metrics_File(file_list)

    # Create csv with HOTA, MOTA, AssA, DetA and IDF1 scores 'model_eval_table_all_matches_for_dupes.csv'
    Create_Table_Metrics()

    # Merge all metrics for combined csv file
    df_model_eval = pd.read_csv('model_eval_table_all_matches_for_dupes.csv')
    df_output = pd.read_csv('output_table_all_matches_for_dupes.csv')
    # Merge based on the 'id' column
    merged_df = pd.merge(df_model_eval, df_output, on='ID', how='inner')  # Use 'how' parameter to specify the type of join
    # Write to csv file
    output_csv_path = 'Full_table_all_matches_for_dupes.csv'
    merged_df.to_csv(output_csv_path, index=False)

    # Create AVG off all csv file 'AVG_table_all_matches_for_dupes.csv'
    df = merged_df
    # Calculate averages for each column except 'ID'
    averages = df.drop(columns=['ID','GT_unique_IDs','Before_RE_ID_unique_IDs','After_RE_ID_unique_IDs']).mean()
    averages_df = averages.to_frame().T
    csv_path = 'AVG_table_all_matches_for_dupes.csv'
    averages_df.to_csv(csv_path, index=False)

