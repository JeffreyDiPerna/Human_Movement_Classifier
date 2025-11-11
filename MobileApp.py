#File to run the GUI

#importing needed libraries
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#Importing external files for the functions
import ExtractFeaturesAndNormalize
import ProcessingData

#Load trained components
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")


#Defining function to process and predict uploaded csv
def process_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    if not file_path:
        return
    try:
        #Loading the CSV data
        df = pd.read_csv(file_path)

        #Extracting ONLY the correct feature columns - CSV MUST HAVE THESE COLUMN LABELS!!
        required_columns = ["Linear Acceleration x (m/s^2)",
                            "Linear Acceleration y (m/s^2)",
                            "Linear Acceleration z (m/s^2)"]

        #print error otherwise
        if not all(col in df.columns for col in required_columns):
            messagebox.showerror("Error", "CSV must contain the correct acceleration columns.")
            return

        #Converting to numpy array
        X_raw = df[required_columns].values

        #Segment data into windows
        windows_x = ProcessingData.split_into_windows(X_raw[:, 0])
        windows_y = ProcessingData.split_into_windows(X_raw[:, 1])
        windows_z = ProcessingData.split_into_windows(X_raw[:, 2])

        #Extracting features from each window, putting into one matrix
        X_features = ExtractFeaturesAndNormalize.extract_features_from_windows(windows_x, windows_y, windows_z)

        #Appling the standardization
        X_scaled = scaler.transform(X_features)

        #Making the predictions and saving results as a CSV
        predictions = model.predict(X_scaled)
        output_file = file_path.replace(".csv", "_predictions.csv")
        pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_file, index=False)
        #print the output
        messagebox.showinfo("Success", f"Predictions saved to {output_file}")

        #Then plot the predictions on graph
        plot_predictions(predictions)

    #error catch in case anything goes wrong
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


#Defined function to plot the predictions
def plot_predictions(predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label="Predictions (0=Walking, 1=Jumping)", marker="o")
    plt.xlabel("Window Index")
    plt.ylabel("Prediction")
    plt.title("Walking vs. Jumping Classification")
    plt.legend()
    plt.show()

#Setup for the GUI
root = tk.Tk()
root.title("Activity Classifier")
root.geometry("400x200")

label = tk.Label(root, text="Upload a CSV file to classify Walking or Jumping", pady=10)
label.pack()

button = tk.Button(root, text="Upload CSV", command=process_file)
button.pack(pady=10)

root.mainloop()
