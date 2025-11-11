#FILE TO VISUALIZE ANY DATASET

#Importing required libraries
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


#Function that visualizes raw and filtered accelerometer data for a given movement type
def visualize_data(df, movement_type):

    #Defining the subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    #Defining any titles and labels
    axis_labels = ['X-Axis', 'Y-Axis', 'Z-Axis']
    raw_columns = ['x', 'y', 'z']
    filtered_columns = ['x_filtered', 'y_filtered', 'z_filtered']

    #For loop to iterate through each axis
    for i in range(3):
        #Raw data plot
        axes[i, 0].plot(df["timestamp"], df[raw_columns[i]], label=f"{axis_labels[i]} (Raw)", color="C" + str(i))
        axes[i, 0].set_title(f"{axis_labels[i]} Raw Accelerometer Data")
        axes[i, 0].set_ylabel("Acceleration (m/s²)")
        axes[i, 0].legend()

        #Filtered data plot
        axes[i, 1].plot(df["timestamp"], df[filtered_columns[i]], label=f"{axis_labels[i]} (Filtered)",
                        color="C" + str(i))
        axes[i, 1].set_title(f"{axis_labels[i]} Filtered Accelerometer Data")
        axes[i, 1].set_ylabel("Acceleration (m/s²)")
        axes[i, 1].legend()

    #Setting the common x-axis label
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")

    plt.suptitle(f"Accelerometer Data for {movement_type}", fontsize=14)
    plt.tight_layout()
    plt.show()

#Function to plot the ROC curve
def plot_roc_curve(model, X_test, y_test):

    #Getting the predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    #Computeing thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    #Computing the AUC score
    auc_score = roc_auc_score(y_test, y_prob)

    #Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")

    #Adding the labels and title
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve for Model Performance")
    plt.legend()
    plt.grid()

    #Final plot
    plt.show()