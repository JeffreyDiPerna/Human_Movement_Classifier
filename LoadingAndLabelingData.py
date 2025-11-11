#FILE FOR LOADING AND LABELING THE CSV FILES

#Importing required libraries
import pandas as pd

#Defining the file paths/movement types so it's easier to use later
file_paths = {
    "Jumping_Trial_One": "Jumping_Trial_One.csv",
    "Jumping_Trial_Two": "Jumping_Trial_Two.csv",
    "Jumping_Trial_Three": "Jumping_Trial_Three.csv",
    "Walking_X_BackPocket": "Walking_X_BackPocket.csv",
    "Walking_X_FrontPocket": "Walking_X_FrontPocket.csv",
    "Walking_X_Hand": "Walking_X_Hand.csv",
    "Walking_XY_BackPocket": "Walking_XY_BackPocket.csv",
    "Walking_XY_FrontPocket": "Walking_XY_FrontPocket.csv",
    "Walking_XY_Hand": "Walking_XY_Hand.csv"
}

#Function to load CSV and add movement label
def load_and_label_data(file_path, movement_type):

    df = pd.read_csv(file_path)

    #Standardize the column names and handle any unexpected spacing
    df.columns = df.columns.str.strip().str.lower()

    #Renaming columns to match expected format
    column_mapping = {
        "time (s)": "timestamp",
        "linear acceleration x (m/s^2)": "x",
        "linear acceleration y (m/s^2)": "y",
        "linear acceleration z (m/s^2)": "z"
    }

    df.rename(columns=column_mapping, inplace=True)

    #Ensuring the required columns exist lol
    required_columns = {"x", "y", "z"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing columns in {file_path}. Found columns: {df.columns}")

    #Add the movement label
    df["movement"] = movement_type
    return df