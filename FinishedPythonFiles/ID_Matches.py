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
import re

def save_frames(video_path, output_folder):

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Initialize frame count
    frame_count = 0

    # Read the first frame
    success, frame = video_capture.read()

    # Loop through the video frames
    while success:
        # Split the filename at the last underscore
        parts = video_path.rsplit("_", 1)
        # The first element after the split will be "129_1"
        desired_part = parts[0]
        # Save the frame as a JPEG image
        output_path = f"{output_folder}/{desired_part}_frame_{frame_count}.jpg"
        cv2.imwrite(output_path, frame)

        # Read the next frame
        success, frame = video_capture.read()

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    video_capture.release()

def Undistorter(folder_path, camera_data_file):
    # Load camera data
    dd = loadmat(camera_data_file)
    distCoeffs1 = deepcopy(dd["distortionCoefficients1"])
    distCoeffs2 = deepcopy(dd["distortionCoefficients2"])
    cameraMatrix1 = deepcopy(dd["intrinsicMatrix1"])
    cameraMatrix2 = deepcopy(dd["intrinsicMatrix2"])
    R = deepcopy(dd["rotationOfCamera2"])
    T = deepcopy(dd["translationOfCamera2"])
    cameraMatrix1[0:2, 2] += 1
    cameraMatrix2[0:2, 2] += 1

#     # Create output folder for undistorted frames
    output_folder = folder_path + "_undistorted"
    os.makedirs(output_folder, exist_ok=True)

    # Loop through images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read image
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            # Undistort image
            img_undistorted = cv2.undistort(img, cameraMatrix1, distCoeffs1)

            # Save undistorted image
            undistorted_filename = os.path.splitext(filename)[0] + '_undistorted.jpg'
            output_path = os.path.join(output_folder, undistorted_filename)
            cv2.imwrite(output_path, img_undistorted)


def epipolar(sorted_files_1,sorted_files_2,camera_data_file,file_path1,file_path2):

    image_left_keypoints_dict = dict()
    image_right_keypoints_dict = dict()
    fundamental_matrix_dict = dict()

    for index in range(len(sorted_files_1)):
        # Target value
        target_value = sorted_files_1[index]

        # Extract frame number from the target value
        target_frame_number = int(re.search(r'_frame_(\d+)_undistorted.jpg', target_value).group(1))

        img_2 = [x for x in sorted_files_2 if ('frame_' + str(target_frame_number) + '_') in x][0]

        img_1 = sorted_files_1[index]

        print(type(img_2))
        print(img_2)
        print(type(img_1))
        print(img_1)

        dd = loadmat(camera_data_file)
        distCoeffs1 = deepcopy(dd["distortionCoefficients1"])
        distCoeffs2 = deepcopy(dd["distortionCoefficients2"])
        cameraMatrix1 = deepcopy(dd["intrinsicMatrix1"])
        cameraMatrix2 = deepcopy(dd["intrinsicMatrix2"])
        R = deepcopy(dd["rotationOfCamera2"])
        T = deepcopy(dd["translationOfCamera2"])
        cameraMatrix1[0:2, 2] += 1
        cameraMatrix2[0:2, 2] += 1
        # Projection matrices
        P1 = np.dot(cameraMatrix1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(cameraMatrix2, np.hstack((R, T.reshape(3, 1))))

        filepath1 = file_path1 + '/' +img_1
        filepath2 = file_path2 + '/'+img_2

        print(filepath1)
        print(filepath2)

        # Load the undistorted stereo images
        img_left = cv2.imread(filepath1, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(filepath2, cv2.IMREAD_GRAYSCALE)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
        keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

        # Undistort keypoints
        keypoints_left_undistorted = cv2.undistortPoints(np.array([kp.pt for kp in keypoints_left]).reshape(-1, 1, 2), cameraMatrix1, distCoeffs1, P=cameraMatrix1)
        keypoints_right_undistorted = cv2.undistortPoints(np.array([kp.pt for kp in keypoints_right]).reshape(-1, 1, 2), cameraMatrix2, distCoeffs2, P=cameraMatrix2)

        # Convert keypoints back to OpenCV KeyPoint format
        keypoints_left_undistorted = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in keypoints_left_undistorted]
        keypoints_right_undistorted = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in keypoints_right_undistorted]

        # Create FLANN matcher object
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match descriptors of left and right images
        matches = flann.knnMatch(descriptors_left, descriptors_right, k=2)

        # Apply ratio test
        good_matches = []
        pts_left = []
        pts_right = []

        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
                pts_left.append(keypoints_left_undistorted[m.queryIdx].pt)
                pts_right.append(keypoints_right_undistorted[m.trainIdx].pt)

        pts_left = np.int32(pts_left)
        pts_right = np.int32(pts_right)
        F, mask = cv2.findFundamentalMat(pts_left,
                                        pts_right,
                                        cv2.FM_LMEDS)

        # We select only inlier points
        pts_left = pts_left[mask.ravel() == 1]
        pts_right = pts_right[mask.ravel() == 1]

        image_left_keypoints_dict[index] = keypoints_left_undistorted
        image_right_keypoints_dict[index] = keypoints_right_undistorted
        fundamental_matrix_dict[index] = F

    return [image_left_keypoints_dict,image_right_keypoints_dict,fundamental_matrix_dict]

def find_matching_points(left_point, fundamental_matrix, right_image_keypoints):

    # Convert left point to homogeneous coordinates
    left_point = np.array([left_point[0], left_point[1], 1])

    # Compute the epipolar line in the right image corresponding to the left point
    epipolar_line = np.dot(fundamental_matrix, left_point)
    # Calculate the bounds for the search area in the right image
    search_area_width = 200  # You can adjust this based on your requirement
    search_area_height = 200  # You can adjust this based on your requirement

    # Define the search area
    search_area = ((left_point[0] - search_area_width, left_point[1] - search_area_height),
                   (left_point[0] + search_area_width, left_point[1] + search_area_height))

    # Initialize a list to store potential matching points
    potential_matching_points = []

    # Iterate through all keypoints in the right image
    for keypoint in right_image_keypoints:
        x, y = keypoint.pt

        # Check if the keypoint lies within the search area
        if search_area[0][0] <= x <= search_area[1][0] and search_area[0][1] <= y <= search_area[1][1]:
            # Compute the distance between the keypoint and the epipolar line
            distance = abs(epipolar_line[0] * x + epipolar_line[1] * y + epipolar_line[2]) / \
                       np.sqrt(epipolar_line[0] ** 2 + epipolar_line[1] ** 2)

            # You can adjust this threshold based on your requirement
            if distance < 30:  # Adjust this threshold as needed
                potential_matching_points.append((x, y))

    return potential_matching_points

def closest_tuple(given_tuple, tuple_list):

    closest = None
    min_distance = float('inf')

    for tpl in tuple_list:
        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(given_tuple, tpl)))
        if distance < min_distance:
            min_distance = distance
            closest = tpl

    return closest

def tuple_differences(tuple1, tuple2):
    if (tuple1 != None) & (tuple2 != None):
        if len(tuple1) != len(tuple2):
            raise ValueError("Tuples must have the same length")

        return tuple(x - y for x, y in zip(tuple1, tuple2))

def Find_Matching_ID(left_id, left_point_tuple, right_view_dataframe, fundamental_matrix, right_image_keypoints):

    matching_points = find_matching_points(left_point_tuple, fundamental_matrix, right_image_keypoints)
#     print(matching_points,"MatchingsPoints")

    options = []

    df = pd.DataFrame(columns=['left_id','right_id','difference','frame'])
    difference = None

    if matching_points != []:

        for index, row in right_view_dataframe.iterrows():
            tuple_right = (row['x'],row['y'])
            if tuple_right != None:
                difference = abs(tuple_differences(tuple_right,closest_tuple(tuple_right, matching_points))[0]) + abs(tuple_differences(tuple_right,closest_tuple(tuple_right, matching_points))[1])
#                 print(difference,"DIF")
                new_row_data = {'left_id': left_id, 'right_id': row['id'],'difference': difference,'frame':row['frame']}
                new_row = pd.DataFrame(new_row_data,index=[0])
                df = pd.concat([df, new_row], ignore_index=True)


        return df
    else:
        return df


def Preds_Maker(df_1,df_2,fundamental_matrix_dict,image_right_keypoints_dict):
    df_preds = pd.DataFrame()
    frames = set(df_1['frame'])
    for frame in frames:
        df_to_use = df_1[df_1['frame'] == frame]
        df_to_check = df_2[df_2['frame'] == frame]

        for index, row in df_to_use.iterrows():
            frame = row['frame']
            id_value = row['id']
            x = row['x']
            y = row['y']
            left_id = id_value
            left_point_tuple = (x,y)
            print(left_point_tuple,frame,left_id)
            if left_point_tuple != None:
                if isinstance(left_point_tuple, tuple):
            #       Find the matching ID for the id_value in this frame
                    df_mathces = Find_Matching_ID(left_id, left_point_tuple, df_to_check, fundamental_matrix_dict[frame], image_right_keypoints_dict[frame])
                    if len(df_mathces) != 0:
                        min_difference_index = df_mathces['difference'].idxmin()

                # Retrieve the corresponding row from the DataFrame
                        row_with_smallest_difference = df_mathces.loc[min_difference_index]

                # Convert the row to a DataFrame
                        row_df = row_with_smallest_difference.to_frame().T

                        df_preds = pd.concat([df_preds,row_df],ignore_index=True)
    return df_preds