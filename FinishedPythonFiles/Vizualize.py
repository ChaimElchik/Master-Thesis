import cv2
import numpy as np
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

from Data_Analysis import *


if __name__ == "__main__":

    # plots for a stereo matched video to visualize results
    # requires to load a df_cor_3d
    df = pd.read_csv('../StereoMatching/df_to_analyze.csv')
    Fish_Trajectories_Graphs(df)
    Fish_Speed_Graphs(df)
    Speed_Acceleration_Df(df)
    Fish_Acceleration_Graphs(df)
    T_Test_Acceleration(df)
    Fish_Path_length_Graph(df)
    AVG_Fish_Path_Length(df)
    Heatmap_of_Fish_Density_Graph(df)
    Temporal_Patterns_Of_Movement(df)
    Fish_speeds_changes(df)
    Spatial_Distribution(df)
    Depth_Analysis(df)


