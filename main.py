#ALL WORK IS ORIGINAL AND DONE BY OUR GROUP

#MAIN FILE TO LOAD PROJECT

#Importing required libraries
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#importing external files
import ExtractFeaturesAndNormalize
import VisualizeData
import LoadingAndLabelingData
import ProcessingData
import MobileApp

#Defining the sampling rate and window size
SAMPLES_PER_SECOND = 50  #Chose 50 for now based on rough calcs, make sure this is consistent
WINDOW_SIZE = SAMPLES_PER_SECOND * 5  #Set to be 5-second windows based on instructions


#Initializing a set of lists to hold all training data and labels
x_train_all = []
y_train_all = []
x_test_all = []
y_test_all = []

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

#Creating the HDF5 file to store all data (as per instructions, raw, filtered, and windowed)
with h5py.File("accelerometer_data.h5", "w") as hdf:
    raw_group = hdf.create_group("raw_data")
    filtered_group = hdf.create_group("preprocessed_data")
    windows_group = hdf.create_group("segmented_data")

    #using a for loop to iterate through each raw csv file / trial to apply processing & storage
    for movement, file_path in file_paths.items():
        print(f"Processing {file_path}...")

        #Load & labeling
        df = LoadingAndLabelingData.load_and_label_data(file_path, movement)


        #Handling missing data
        df = ProcessingData.handle_missing_data(df)

        #Applying the filter on raw data
        df = ProcessingData.apply_filter(df)

        #Convert files to NumPy
        x_data, y_data, z_data = df["x_filtered"].values, df["y_filtered"].values, df["z_filtered"].values

        #Segmenting each one of axis into windows
        x_windows = ProcessingData.split_into_windows(x_data)
        y_windows = ProcessingData.split_into_windows(y_data)
        z_windows = ProcessingData.split_into_windows(z_data)

        #storing the extracted features into one matrix (X)
        X_features = ExtractFeaturesAndNormalize.extract_features_from_windows(x_windows, y_windows, z_windows)

        #Droping the non-numeric columns (labels) and only keeping the numbers before saving to HDF5
        df_numeric = df.select_dtypes(include=[np.number])

        #Save the numeric data
        raw_group.create_dataset(movement, data=df_numeric.to_numpy())

        #Saving the movement labels as a fixed-length string dataset and byte strings
        movement_labels = np.array(df["movement"], dtype="S10")
        raw_group.create_dataset(f"{movement}_labels", data=movement_labels)

        #Storing the filtered data within the HDF5 File
        filtered_group.create_dataset(f"{movement}_filtered", data=df[["x_filtered", "y_filtered", "z_filtered"]].to_numpy())


        #Print to make sure for debugging
        print(f" {file_path} stored successfully!")

        #Feature matrix (X) - Excluding the label column
        if 'movement' in df.columns:
            X = df.drop('movement', axis=1).values
        else:
            X = df.values  # Fallback in case 'movement' doesn't exist

        #Converting labels ('walking'/'jumping') to binary (0 for 'walking' and 1 for 'jumping')
        df_labels = df['movement'].apply(lambda x: 1 if x.startswith('Jumping') else 0).values

        #Segmenting labels to match the windows
        num_windows = len(x_windows)

        #Creating an array for the segmented labels to append onto
        segmented_labels = []

        # Visualizing the data
        VisualizeData.visualize_data(df, movement)

        #Assuming each window corresponds to the first label in the window in the data (like each trial is only Walking or Jumping exclusively)
        for i in range(num_windows):
            #Use the first label in the window
            segmented_labels.append(df_labels[i * WINDOW_SIZE])

        #converting to a numpy array
        segmented_labels = np.array(segmented_labels)

        #Debugging shape mismatch (should match)
        #print(f"X_windows_flat shape: {X_features.shape}")
        #print(f"segmented_labels shape: {segmented_labels.shape}")


        #Splitting the required data into training and test sets - 90% for training, 10% for testing, and making sure there is no overlap, using 42 as seed
        x_train, x_test, y_train, y_test = train_test_split(X_features, segmented_labels, test_size=0.1, random_state=42, stratify=segmented_labels)


        #Collecting all training data and labels (for train and test sets)
        x_train_all.append(x_train)
        y_train_all.append(y_train)
        x_test_all.append(x_test)
        y_test_all.append(y_test)

    # Creating groups for training and test data inside windows_group
    train_group = windows_group.create_group("train_data")
    test_group = windows_group.create_group("test_data")

    train_group.create_dataset("X_train", data=np.vstack(x_train_all))  # Stack vertically
    train_group.create_dataset("y_train", data=np.hstack(y_train_all))  # Stack horizontally
    test_group.create_dataset("X_test", data=np.vstack(x_test_all))  # Stack vertically
    test_group.create_dataset("y_test", data=np.hstack(y_test_all))  # Stack horizontally

#Now going through and reading the created datasets
with h5py.File("accelerometer_data.h5", "r") as hdf:
    #printing everything created for testing purposes
    print("Raw Data Stored:", list(hdf["raw_data"].keys()))
    print("Filtered Data Stored:", list(hdf["preprocessed_data"].keys()))
    print("Windowed Data Stored:", list(hdf["segmented_data"].keys()))


#Concatenating all training and test data post loop
#Combine all feature and label arrays - Train
x_train_all = np.concatenate(x_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0)
#Combine all feature and label arrays - Test
x_test_all = np.concatenate(x_test_all, axis=0)
y_test_all = np.concatenate(y_test_all, axis=0)

#Standardizing the features using imported library
# print(f"Shape of X_train: {x_train_all.shape}") --> FOR DEBUG
# print(f"Shape of X_test: {y_test_all.shape}")
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_all)
x_test_scaled = scaler.transform(x_test_all)


#Training the final model (setting a max iterations limit to be 10,000)
model = LogisticRegression(max_iter=10000)

#Fitting the model with the trained data
model.fit(x_train_scaled, y_train_all)

#Evaluating the model on the combined training data
y_test_prediction = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test_all, y_test_prediction)

#printing accuracy
print(f"Model trained successfully! Accuracy on training set: {accuracy:.2f}")

VisualizeData.plot_roc_curve(model,x_test_scaled, y_test_all);

#Need to save the scaler, pca transformation, and trained model for the GUI
joblib.dump(scaler, "scaler.pkl")
#joblib.dump(pca, "pca.pkl")
joblib.dump(model, "model.pkl")

#print statement to ensure complete
print("Model and preprocessors saved successfully!")

