#FILE TO EXTRACT FEATURES OF DATA AND THEN NORMALIZE

#Importing required libraries
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

#Function to extract features from a single window of data (x, y, z)
def extract_features_from_window(window_data):
    # Calculate features for each axis
    features = {
        "max_x": np.max(window_data[:, 0]),
        "min_x": np.min(window_data[:, 0]),
        "range_x": np.ptp(window_data[:, 0]),
        "mean_x": np.mean(window_data[:, 0]),
        "median_x": np.median(window_data[:, 0]),
        "variance_x": np.var(window_data[:, 0]),
        "skewness_x": skew(window_data[:, 0]),
        "kurtosis_x": kurtosis(window_data[:, 0]),
        "rms_x": np.sqrt(np.mean(window_data[:, 0] ** 2)),
        "pr_ratio_x": np.max(window_data[:, 0]) / np.sqrt(np.mean(window_data[:, 0] ** 2)),

        "max_y": np.max(window_data[:, 1]),
        "min_y": np.min(window_data[:, 1]),
        "range_y": np.ptp(window_data[:, 1]),
        "mean_y": np.mean(window_data[:, 1]),
        "median_y": np.median(window_data[:, 1]),
        "variance_y": np.var(window_data[:, 1]),
        "skewness_y": skew(window_data[:, 1]),
        "kurtosis_y": kurtosis(window_data[:, 1]),
        "rms_y": np.sqrt(np.mean(window_data[:, 1] ** 2)),
        "pr_ratio_y": np.max(window_data[:, 1]) / np.sqrt(np.mean(window_data[:, 1] ** 2)),

        "max_z": np.max(window_data[:, 2]),
        "min_z": np.min(window_data[:, 2]),
        "range_z": np.ptp(window_data[:, 2]),
        "mean_z": np.mean(window_data[:, 2]),
        "median_z": np.median(window_data[:, 2]),
        "variance_z": np.var(window_data[:, 2]),
        "skewness_z": skew(window_data[:, 2]),
        "kurtosis_z": kurtosis(window_data[:, 2]),
        "rms_z": np.sqrt(np.mean(window_data[:, 2] ** 2)),
        "pr_ratio_z": np.max(window_data[:, 2]) / np.sqrt(np.mean(window_data[:, 2] ** 2)),
    }

    #Converting the features to a single row
    return features

#Function to extract features from all windows
def extract_features_from_windows(windows_x, windows_y, windows_z):
    feature_list = []

    for i in range(len(windows_x)):
        #Combininh data from the three axes into a single array
        window_data = np.stack((windows_x[i], windows_y[i], windows_z[i]), axis=1)
        #Extracting features for the current window
        features = extract_features_from_window(window_data)
        feature_list.append(features)

    return pd.DataFrame(feature_list)

#Normalize the features using Z-score standardization
def normalize_features(df):
    return (df - df.mean())/df.std()
