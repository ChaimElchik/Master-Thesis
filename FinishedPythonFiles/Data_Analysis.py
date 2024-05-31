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
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
import seaborn as sns



# Assuming your dataframe is named 'df'


def Fish_Trajectories_Graphs(df):
    # Group the dataframe by fish_id
    grouped = df.groupby('fish_id')

    # Iterate over each group (each fish ID) and create a separate plot
    for fish_id, group in grouped:
        # Create a new figure for each fish ID
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory of the current fish ID
        ax.plot(group['x'], group['y'], group['z'], label=f'Fish ID: {fish_id}')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Fish ID: {fish_id} Trajectory')

        # Add a legend
        ax.legend()

        # Show the plot
        end_name = str(fish_id) + "Fish_Trajectory_plot.png"
        plt.savefig(end_name)


def Fish_Speed_Graphs(df):
    # Calculate speed
    df['speed'] = ((df['x'].diff()**2 + df['y'].diff()**2 + df['z'].diff()**2) ** 0.5).fillna(0)

    # Plot speed over time for each fish ID
    for fish_id, group in df.groupby('fish_id'):
        plt.plot(group['frame'], group['speed'], label=f'Fish ID: {fish_id}')
        plt.xlabel('Frame')
        plt.ylabel('Speed')
        plt.title(f'Speed Analysis - Fish ID: {fish_id}')
        plt.legend()
        end_name = str(fish_id) + "plot.png"
        plt.savefig(end_name)

def Speed_Acceleration_Df(df):
    # Calculate speed per fish per frame
    df['speed'] = ((df.groupby('fish_id')['x'].diff()**2 + df.groupby('fish_id')['y'].diff()**2 + df.groupby('fish_id')['z'].diff()**2) ** 0.5).fillna(0)

    # Calculate acceleration per fish per frame
    df['acceleration'] = df.groupby('fish_id')['speed'].diff().fillna(0)

    # Pair speed and acceleration together
    speed_acceleration_pairs = df[['fish_id', 'frame', 'speed', 'acceleration']]

    end_name = "Speed_Acceleration_Df.csv"
    speed_acceleration_pairs.to_csv(end_name, index=False)

def Fish_Acceleration_Graphs(df):
    # Calculate acceleration
    df['acceleration'] = ((df['x'].diff().diff()**2 + df['y'].diff().diff()**2 + df['z'].diff().diff()**2) ** 0.5).fillna(0)

    # Plot acceleration over time for each fish ID
    for fish_id, group in df.groupby('fish_id'):
        plt.plot(group['frame'], group['acceleration'], label=f'Fish ID: {fish_id}')
        plt.xlabel('Frame')
        plt.ylabel('Acceleration')
        plt.title(f'Acceleration Analysis - Fish ID: {fish_id}')
        plt.legend()
        end_name = str(fish_id) + "plot.png"
        plt.savefig(end_name)


def T_Test_Acceleration(df):
    # Calculate summary statistics of acceleration for each fish ID
    acceleration_stats = df.groupby('fish_id')['acceleration'].describe()
    # Investigate differences in acceleration patterns between fish IDs using statistical tests
    from scipy.stats import ttest_ind
    with open("acceleration_analysis.txt", "w") as f:
        f.write("Summary statistics of acceleration for each fish ID:\n")
        f.write(acceleration_stats.to_string())  # Write stats to file



def Fish_Path_length_Graph(df):
    # Calculate path length for each fish ID
    path_lengths = df.groupby('fish_id').apply(lambda group: ((group['x'].diff()**2 + group['y'].diff()**2 + group['z'].diff()**2) ** 0.5).sum())

    # Plot path length for each fish ID
    plt.bar(path_lengths.index, path_lengths.values)
    plt.xlabel('Fish ID')
    plt.ylabel('Path Length')
    plt.title('Path Length Analysis')
    end_name = "Fish_Path_Length_Plot.png"
    plt.savefig(end_name)

def AVG_Fish_Path_Length(df):
    # Calculate total path length covered by each fish ID
    path_lengths = df.groupby('fish_id').apply(lambda group: ((group['x'].diff()**2 + group['y'].diff()**2 + group['z'].diff()**2) ** 0.5).sum())

    # Compare path lengths between different fish IDs using descriptive statistics
    mean_path_length = str(path_lengths.mean())
    with open("AVG_Fish_Path_Length.txt", "w") as f:
        f.write("Average Path Length:\n")
        f.write(mean_path_length)  # Write stats to file

def Heatmap_of_Fish_Density_Graph(df):
    plt.figure()  # Create a new figure to avoid conflicts
    plt.clf()  # Clear the current figure
    # Create a 2D histogram (heatmap) of fish density
    plt.hist2d(df['x'], df['y'], bins=50, cmap='hot')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of Fish Density')
    end_name = "Heatmap_of_Fish_Density_Graph.png"
    plt.savefig(end_name)
    plt.close()  # Close the figure to free memory


def Temporal_Patterns_Of_Movement(df):
    # Define the total number of frames in the video
    total_frames = df['frame'].max() + 1

    # Calculate temporal patterns of movement
    # Convert frame to time in seconds
    df['time_seconds'] = df['frame'] * (74 / total_frames)  # Assuming the total video length is 74 seconds

    # Group data by time intervals (e.g., seconds) and calculate mean speed for each interval
    time_interval_seconds = 10  # Define the time interval in seconds
    df['time_interval'] = (df['time_seconds'] // time_interval_seconds) * time_interval_seconds
    mean_speed_by_interval = df.groupby('time_interval')['speed'].mean()

    # Plot mean speed over time
    plt.plot(mean_speed_by_interval.index, mean_speed_by_interval.values)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mean Speed')
    plt.title('Temporal Patterns of Movement')
    end_name = "Temporal_Patterns_Of_Movement_Graph.png"
    plt.savefig(end_name)

def Fish_speeds_changes(df):
    # Calculate speed difference compared to the previous frame for each fish
    df['speed_diff'] = df.groupby('fish_id')['speed'].diff()

    # Find frames where speed increases for each fish
    increased_speed_frames = df[df['speed_diff'] > 0]

    import pandas as pd

    # Assuming 'df' is your dataframe containing speed data per fish per frame

    # Calculate speed difference compared to the previous frame for each fish
    df['speed_diff'] = df.groupby('fish_id')['speed'].diff()

    # Define a threshold for significant speed increase
    threshold = 0.5  # Adjust as needed

    # Find frames where speed increases more than the threshold for each fish
    increased_speed_frames = df[df['speed_diff'] > threshold]

    # Print the frames where speed increases more than the threshold for each fish
    with open("Fish_speeds_changes.txt", "w") as f:
        for fish_id, group in increased_speed_frames.groupby('fish_id'):
            f.write(f"Fish ID {fish_id}: Frames with speed increase > {threshold} - {group['frame'].tolist()}\n")  # Write stats to file

def Spatial_Distribution(df):
    plt.figure()  # Create a new figure to avoid conflicts
    plt.clf()  # Clear the current figure
    # Create a new 2D axis
    ax = plt.gca()
    # Visualize spatial distribution of fish within the tank
    sns.scatterplot(x='x', y='y', data=df, hue='fish_id', ax=ax)
    end_name = "Spatial_Distribution_Graph.png"
    plt.savefig(end_name)
    plt.close()  # Close the figure to free memory


def Depth_Analysis(df):

    # Calculate average depth per fish per frame
    average_depth_per_fish = df.groupby('fish_id')['z'].mean()

    # Calculate depth range per fish
    depth_range_per_fish = df.groupby('fish_id')['z'].max() - df.groupby('fish_id')['z'].min()

    # Calculate depth variability per fish (e.g., standard deviation)
    depth_variability_per_fish = df.groupby('fish_id')['z'].std()

    # Visualize depth trajectories for each fish
    for fish_id, group in df.groupby('fish_id'):
        plt.plot(group['frame'], group['z'], label=f'Fish {fish_id}')

    plt.xlabel('Frame')
    plt.ylabel('Depth')
    plt.title('Depth Trajectories for Each Fish')
    plt.legend()
    end_name = "Depth_Analysis_Graph.png"
    plt.savefig(end_name)
