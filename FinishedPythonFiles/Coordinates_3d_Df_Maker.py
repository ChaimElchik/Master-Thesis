
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


def cor_maker_3d(df1,df2,camera_data_file,gt_matches):

    dd = loadmat(camera_data_file)
    distCoeffs1 = deepcopy(dd["distortionCoefficients1"])
    distCoeffs2 = deepcopy(dd["distortionCoefficients2"])
    cameraMatrix1 = deepcopy(dd["intrinsicMatrix1"])
    cameraMatrix2 = deepcopy(dd["intrinsicMatrix2"])
    R = deepcopy(dd["rotationOfCamera2"])
    T = deepcopy(dd["translationOfCamera2"])
    cameraMatrix1[0:2, 2] += 1
    cameraMatrix2[0:2, 2] += 1
    P1 = np.dot(cameraMatrix1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(cameraMatrix2, np.hstack((R, T.reshape(3, 1))))

    df_3d_v2_ground_truth = pd.DataFrame()
    for frame in set(df1['frame']):
    #     print(frame,"frame")
        df1_temp = df1.loc[df1['frame'] == frame]
        df2_temp = df2.loc[df2['frame'] == frame]
        for cur_id in list(df1_temp['id']):
            df1_tt = df1_temp.loc[df1_temp['id'] == cur_id]
    #         df2_tt = df2_temp.loc[df2_temp['id'] == cur_id]
            x_1 = float(df1_tt['x'])
            y_1 = float(df1_tt['y'])
            match_id = gt_matches[cur_id]
    #         print(cur_id,"cur",(match_id),"matchID")
            if match_id in list(df2_temp['id']):
    #             print(cur_id,match_id, "pair")
                df2_tt = df2_temp.loc[df2_temp['id'] == match_id]
                x_2 = float(df2_tt['x'])
                y_2 = float(df2_tt['y'])
    #             print(x_1,y_1,cur_id,l_id_corrected,match_id,x_2,y_2)
                left_cor = np.array([x_1,y_1])
                right_cor = np.array([x_2,y_2])

                undistortedPoints1 = cv2.undistortPoints(left_cor, cameraMatrix1, distCoeffs1, P=cameraMatrix1)  # result Nx1x2. No need for np.expand_dims(imagePoints1, axis=1)
                undistortedPoints2 = cv2.undistortPoints(right_cor, cameraMatrix2, distCoeffs2, P=cameraMatrix2)

                # Perform triangulation
                point_3d_homogeneous = cv2.triangulatePoints(P1, P2, undistortedPoints1, undistortedPoints2)
                # Convert homogeneous coordinates to Cartesian
                point_3d_cartesian = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
                # Extract x, y, z coordinates
                x, y, z = point_3d_cartesian.flatten()

            #     new_row_data = {'left_id': left_id, 'right_id': row['id'],'difference': difference,'frame':row['frame']}
            #     new_row = pd.DataFrame(new_row_data,index=[0])

                df_temp = pd.DataFrame({'frame':frame,'left_id': cur_id, 'right_id': match_id,'x': x,'y': y, 'z': z},index=[0])
                df_3d_v2_ground_truth = pd.concat([df_3d_v2_ground_truth, df_temp], ignore_index=True)

    return df_3d_v2_ground_truth