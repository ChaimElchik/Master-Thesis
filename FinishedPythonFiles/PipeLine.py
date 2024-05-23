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

from Mot_Cleaner import clean_file
from Detection_Video_Creator import TR_Vid_Creator
from Yolov8_Data_Prep import prepare_data_for_yolov8
from Deploy_Model import Deploy_Model
from Re_ID_Tracks import *
from ID_Matches import *
from ID_Match_DF_And_Dict import *
from Coordinates_3d_Df_Maker import *


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

    # videos cam 1 list
    video_list_cam_1 = [
        '8_1.mp4', '10_1.mp4',  '13_1.mp4', '14_1.mp4',
        '15_1.mp4', '16_1.mp4',  '20_1.mp4',  '23_1.mp4',
        '24_1.mp4',  '26_1.mp4',  '29_1.mp4',  '32_1.mp4',
        '33_1.mp4',  '41_1.mp4',  '42_1.mp4',  '43_1.mp4',
        '46_1.mp4',  '48_1.mp4',  '51_1.mp4',  '54_1.mp4',
        '58_1.mp4',  '60_1.mp4',  '61_1.mp4',  '62_1.mp4',
        '70_1.mp4',  '83_1.mp4',  '92_1.mp4', '116_1.mp4',
        '122_1.mp4',  '129_1.mp4','134_1.mp4',  '143_1.mp4','161_1.mp4', '168_1.mp4',
        '175_1.mp4', '183_1.mp4','186_1.mp4',  '187_1.mp4',  '189_1.mp4',
        '191_1.mp4',  '194_1.mp4', '195_1.mp4', '198_1.mp4',
        '202_1.mp4',  '204_1.mp4', '206_1.mp4',  '212_1.mp4',
        '214_1.mp4',  '217_1.mp4',  '223_1.mp4', '231_1.mp4', '234_1.mp4',
        '236_1.mp4', '239_1.mp4',  '240_1.mp4',  '242_1.mp4',
        '244_1.mp4', '247_1.mp4',  '249_1.mp4',  '260_1.mp4',
        '261_1.mp4',  '265_1.mp4',  '268_1.mp4',  '270_1.mp4',
        '278_1.mp4',  '280_1.mp4',  '283_1.mp4',  '289_1.mp4',
        '294_1.mp4',  '295_1.mp4', '296_1.mp4',  '297_1.mp4',
        '301_1.mp4',  '302_1.mp4',  '303_1.mp4',  '304_1.mp4',
        '308_1.mp4',  '309_1.mp4',  '312_1.mp4',  '349_1.mp4',
        '368_1.mp4',  '371_1.mp4',  '373_1.mp4',  '374_1.mp4',
        '376_1.mp4',  '378_1.mp4',  '384_1.mp4',  '392_1.mp4',  '406_1.mp4',
        '432_1.mp4',  '437_1.mp4',
    ]

    file_list_cam_1 = [filename.replace('.mp4', '_tr.csv') for filename in video_list]

    # clean mot files
    for file in file_list:
        path = '../mots/'+file
        clean_file(path)

    # Example usage add list of videos for training and matching mot files
    video_paths_train = ['../vids/349_2.mp4','../vids/406_1.mp4', '../vids/161_2.mp4']
    csv_paths_train = ['349_2_clean.txt','406_1_clean.txt','161_2_clean.txt']

    # name output directory
    output_dir_train = 'dataset_training'
    prepare_data_for_yolov8(video_paths_train, csv_paths_train, output_dir_train)

    # Example usage add list of videos for validation and matching mot files
    video_paths_val = ['../vids/129_2.mp4','../vids/8_1.mp4', '../vids/32_2.mp4']
    csv_paths_val = ['129_2_clean.txt','8_1_clean.txt','32_2_clean.txt']

    # name output directory
    output_dir_val = 'dataset_val'
    prepare_data_for_yolov8(video_paths_val, csv_paths_val, output_dir_val)

    # Train Model, assume that folders are in correct format and data.yaml file exists.
    # results = model.train(data="data.yaml", epochs=20, imgsz=640)

    # laod model
    model = YOLO('det_best_bgr29.pt')

    # Deploy model on videos
    Deploy_Model(video_list,model)

    # Re-ID Fish save new tracks
    for file_name in file_list:
        tr_path = 'df' + file_name
        df_TR = pd.read_csv(tr_path)
        df_TR = (Re_ID(df_TR))
        parts = file_name.split('_')
        extracted_part = parts[0] + '_' + parts[1]
        end_name = 'df_' + extracted_part  + '_Re_ID.csv'
        df_TR.to_csv(end_name, index=False)

    for file_name in file_list:
        parts = file_name.split('_')
        file_num = parts[0] + '_' + parts[1]
        # Read the video file
        input_video_path = 'vids/' + file_num + '.mp4'
        # Read the bounding box coordinates from the CSV file
        csv_path = 'df_' + file_num + '_Re_ID.csv'
        df_detection = pd.read_csv(csv_path)
        output_video_name = file_num + '_Detection.mp4'
        TR_Vid_Creator(input_video_path,output_video_name,df_detection)

    for file_name in file_list:
        parts = file_name.split('_')
        extracted_part = parts[0] + '_' + parts[1]
        video_path = '../vids/' + extracted_part + '.mp4'
        output_folder = "Frames_" + extracted_part
        save_frames(video_path, output_folder)
        if extracted_part < 42:
            camera_data_file = "stereoParams_Dep1.mat"
        elif 42 <= extracted_part < 116:
            camera_data_file = "stereoParams_Dep2.mat"
        elif 116 <= extracted_part < 168:
            camera_data_file = "stereoParams_Dep4.mat"
        elif 168 <= extracted_part < 183:
            camera_data_file = "stereoParams_Dep5.mat"
        elif 183 <= extracted_part < 223:
            camera_data_file = "stereoParams_Dep6.mat"
        elif 223 <= extracted_part < 268:
            camera_data_file = "stereoParams_Dep7.mat"
        elif 268 <= extracted_part < 296:
            camera_data_file = "stereoParams_Dep8.mat"
        elif 296 <= extracted_part < 328:
            camera_data_file = "stereoParams_Dep9.mat"
        elif 328 <= extracted_part < 376:
            camera_data_file = "stereoParams_Dep10.mat"
        elif 376 <= extracted_part < 406:
            camera_data_file = "stereoParams_Dep11.mat"
        elif extracted_part >= 406:
            camera_data_file = "stereoParams_Dep12.mat"

        Undistorter(output_folder, camera_data_file)

    for file_name in file_list:
        parts = file_name.split('_')
        extracted_part = parts[0] + '_' + parts[1]
        # Define the folder path
        folder_path_1 = 'Frames_' + extracted_part + '_undistorted'
        # List all files in the folder
        file_names_1 = os.listdir(folder_path_1)
        # Extract the frame numbers from the file names and sort
        sorted_files_1 = sorted([file_name for file_name in file_names_1 if file_name != '.DS_Store'], key=lambda x: int(x.split('_frame_')[1].split('_undistorted.jpg')[0]))
        num = extracted_part.split('_')[0]
        # Define the folder path
        folder_path_2 = 'Frames_' + num + '2_undistorted'
        # List all files in the folder
        file_names_2 = os.listdir(folder_path_2)
        # Extract the frame numbers from the file names
        frame_numbers_2 = [int(file_name.split('_frame_')[1].split('_undistorted.jpg')[0]) for file_name in file_names_2]
        # Sort the file names based on the frame numbers
        sorted_files_2 = [file_name for _, file_name in sorted(zip(frame_numbers_2, file_names_2))]
        filepath1 = folder_path_1 + '/'
        filepath2 = folder_path_2 + '/'
        # run epiplor for video pair
        [image_left_keypoints_dict,image_right_keypoints_dict,fundamental_matrix_dict] = epipolar(sorted_files_1,sorted_files_2,camera_data_file,filepath1,filepath2)
        path1 = 'df_' + extracted_part + '_Re_ID.csv'
        path2 = 'df_' + num + '_2_Re_ID.csv'
        df_1 = pd.read_csv(path1)
        df_2 = pd.read_csv(path2)
        # create dataframe with maches per fish per frame
        df_preds = Preds_Maker(df_1,df_2,fundamental_matrix_dict,image_right_keypoints_dict)
        output_name = 'df_preds_Re_ID_' + num +'.csv'
        df_preds.to_csv(output_name, index=False)
        # create and save csv files with best matches per fish for the pair of videos
        Match_DF_Maker(extracted_part)
        path_name_df_most_common = 'most_common_' + extracted_part + '.csv'
        # create dictionary for best matches from csv file
        left_id_to_right_id = Match_Dict_Maker(path_name_df_most_common)
        # create df with 3d coordinates
        df_cor_3d = cor_maker_3d(df_1,df_2,camera_data_file,left_id_to_right_id)
        df_cor_3d = df_cor_3d .drop('right_id',axis=1)
        df_cor_3d = df_cor_3d .rename(columns={'left_id': 'fish_id'})
        # save df with 3d coordinates
        end_name = 'df_3d' + num  + '.csv'
        df_cor_3d .to_csv(end_name, index=False)



