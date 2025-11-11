#FILE FOR PROCESSING THE DATA

#Importing required libraries
import numpy as np

#Defining the sampling rate and window size
SAMPLES_PER_SECOND = 50  #Chose 50 for now based on rough calcs, make sure this is consistent
WINDOW_SIZE = SAMPLES_PER_SECOND * 5  #Set to be 5-second windows based on instructions

#Function to handle missing values by applying interpolation only on the numeric columns
def handle_missing_data(df):

    #Converting all columns to numeric where possible, force errors to identify issues
    df = df.infer_objects(copy=False) #had an issue with versions here
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    #Interpolating and fill NaNs for numeric columns only
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear").fillna(df[numeric_cols].mean())

    return df

#Function that applies a moving average filter to smooth data
def moving_average(data, window_size=5):
    return data.rolling(window=window_size, center=True, min_periods=1).mean()

#Function that applies filtering to each axis
def apply_filter(df):
    df["x_filtered"] = moving_average(df["x"])
    df["y_filtered"] = moving_average(df["y"])
    df["z_filtered"] = moving_average(df["z"])
    return df

#Function that Splits time-series data into fixed-size windows
def split_into_windows(data, window_size=WINDOW_SIZE):
    num_windows = len(data) // window_size
    return np.array(np.split(data[:num_windows * window_size], num_windows))