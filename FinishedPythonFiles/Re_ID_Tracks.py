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


def find_id_transitions(df, frame_threshold=100):
    # Dictionary to store the last position of each ID
    last_positions = {}

    # Set to store unique ID transitions
    unique_id_transitions = set()

    for index, row in df.iterrows():
        frame = row['frame']
        id_value = row['id']
        x = row['x']
        y = row['y']

        if id_value in last_positions:
            # If ID already exists in last_positions
            last_frame, last_x, last_y = last_positions[id_value]

            if frame - last_frame > frame_threshold:
                # ID has disappeared for more than frame_threshold frames
                last_positions.pop(id_value)
        else:
            # ID is not in last_positions
            for prev_id, (prev_frame, prev_x, prev_y) in last_positions.items():
                distance = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
                if distance < 50 and frame - prev_frame < frame_threshold:
                    # New ID appeared close to the position of a disappeared ID
                    # Check if this transition has already been found
                    if (prev_id, id_value) not in unique_id_transitions:
                        unique_id_transitions.add((prev_id, id_value))

        # Update last position for the current ID
        last_positions[id_value] = (frame, x, y)

    return list(unique_id_transitions)

def filter_tuples_by_first_value(tuples_list, values_list):
    filtered_tuples = []
    for pair in tuples_list:
        first_value, second_value = pair
        if first_value in values_list and second_value not in values_list:
            filtered_tuples.append(pair)
    return filtered_tuples

def remove_tuples_with_large_time_gap(tuples_list, df):
    filtered_tuples = []
    for pair in tuples_list:
        first_id, second_id = pair
        first_id_last_frame = df[df['id'] == first_id]['frame'].max()
        second_id_first_frame = df[df['id'] == second_id]['frame'].min()

        if abs(first_id_last_frame - second_id_first_frame) < 100 :
            filtered_tuples.append(pair)

    return filtered_tuples

def replace_ids_in_dataframe(tuples_list, df):
    for pair in tuples_list:
        first_id, second_id = pair
        df.loc[df['id'] == second_id, 'id'] = first_id

    return df


def renumber_ids(df):
    # Get unique IDs in the DataFrame
    unique_ids = df['id'].unique()

    # Create a mapping from old IDs to new IDs
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

    # Replace old IDs with new IDs in the DataFrame
    df['id'] = df['id'].map(id_mapping)

    return df

def Re_ID(df):

#     Step 1
    less_than_5_list = []

    # Group by 'ID' and count unique frames each ID appears in
    id_counts = df.groupby('id')['frame'].nunique()

    # Filter IDs that appear in less than 5 frames
    ids_less_than_5_frames = id_counts[id_counts < 30]

    # Create a dictionary to store the corresponding frames for each ID
    id_frames_dict = {}

    # Iterate over the IDs and find the corresponding frames
    for id_ in ids_less_than_5_frames.index:
        frames = df[df['id'] == id_]['frame'].unique().tolist()
        id_frames_dict[id_] = frames

    for id_, frames in id_frames_dict.items():
        less_than_5_list.append(id_)

    # List of IDs to remove
    ids_to_remove = less_than_5_list

    # Removing rows with IDs from the list
    df = df[~df['id'].isin(ids_to_remove)]

#     Step 2

    begin_ids = list(df['id'].loc[df['frame'] == 0])

    # Given list of IDs
    given_ids = begin_ids

    not_untill_end = []

    last_frame_appearance = {}

    # Iterate over the IDs and find the last frame where each ID appears
    for id in given_ids:
        last_frame = df[df['id'] == id]['frame'].max()
        last_frame_appearance[id] = last_frame if not pd.isnull(last_frame) else "Does not appear"
        if last_frame != df['frame'].max():
            not_untill_end.append(id)

    id_transitions = find_id_transitions(df, frame_threshold=100)

    filtered_tuples = filter_tuples_by_first_value(id_transitions, not_untill_end)

#     Step 3

    # check if second id in tuple appears 40 frames before last frame of first id in tuple,
    # if so then probably just two fish close to each other

    filtered_tuples = remove_tuples_with_large_time_gap(filtered_tuples, df)

#     Step 4


    # Create a dictionary where keys are the second elements and values are lists of corresponding first elements
    second_to_first_dict = {}
    for first, second in filtered_tuples:
        if second not in second_to_first_dict:
            second_to_first_dict[second] = [first]
        else:
            second_to_first_dict[second].append(first)

    # Check if any value has more than one occurrence
    for second, first_list in second_to_first_dict.items():
        if len(first_list) > 1:
            for first in first_list:
                # Check if the corresponding first element appears in another tuple
                if any((first, other_second) in filtered_tuples for other_second in second_to_first_dict if other_second != second):
#                     print("Tuple with second element", second, "and first element", first, "meets the condition")

                    # Tuple to remove
                    tuple_to_remove = (first, second)

                    # Remove the tuple from the tuple list
                    filtered_tuples = [pair for pair in filtered_tuples if pair != tuple_to_remove]

#     Step 5

    ids_to_remove = [x[1] for x in filtered_tuples]

    df = replace_ids_in_dataframe(filtered_tuples, df)

#     Step 6
    # check for ids from list that dissapear and for ids that appear around the same frame the previous one
    # disapeard and remain for atleast 500 more frames or untill the end

    while filtered_tuples != []:

#     Step 2

        begin_ids = list(set(df['id']))

        # Given list of IDs
        given_ids = begin_ids

        not_untill_end = []

        last_frame_appearance = {}

        # Iterate over the IDs and find the last frame where each ID appears
        for id in given_ids:
            last_frame = df[df['id'] == id]['frame'].max()
            last_frame_appearance[id] = last_frame if not pd.isnull(last_frame) else "Does not appear"
            if last_frame != df['frame'].max():
                not_untill_end.append(id)

        id_transitions = find_id_transitions(df, frame_threshold=100)

        filtered_tuples = filter_tuples_by_first_value(id_transitions, not_untill_end)

        #     Step 3
        # check if second id in tuple appears 40 frames before last frame of first id in tuple,
        # if so then probably just two fish close to each other

        filtered_tuples = remove_tuples_with_large_time_gap(filtered_tuples, df)

    #     Step 4
        # Create a dictionary where keys are the second elements and values are lists of corresponding first elements
        second_to_first_dict = {}
        for first, second in filtered_tuples:
            if second not in second_to_first_dict:
                second_to_first_dict[second] = [first]
            else:
                second_to_first_dict[second].append(first)

        # Check if any value has more than one occurrence
        for second, first_list in second_to_first_dict.items():
            if len(first_list) > 1:
                for first in first_list:
                    # Check if the corresponding first element appears in another tuple
                    if any((first, other_second) in filtered_tuples for other_second in second_to_first_dict if other_second != second):

                        # Tuple to remove
                        tuple_to_remove = (first, second)

                        # Remove the tuple from the tuple list
                        filtered_tuples = [pair for pair in filtered_tuples if pair != tuple_to_remove]

    #     Step 5

        ids_to_remove = [x[1] for x in filtered_tuples]

        df = replace_ids_in_dataframe(filtered_tuples, df)

    renumbered_df = renumber_ids(df.copy())

    return renumbered_df

