import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_mean_erp(trial_points_file, ecog_data_file):
    # Read the CSV files
    trial_points = pd.read_csv(trial_points_file, header=None, names=["start", "peak", "finger"])
    ecog_data = pd.read_csv(ecog_data_file, header=None, names=["ecog"])

    # Convert data to int type
    trial_points["start"] = trial_points["start"].astype(int)
    trial_points["peak"] = trial_points["peak"].astype(int)

    # Initialize a matrix of 5x1201 (5 fingers, each with 1201 data points)
    fingers_erp_mean = np.zeros((5, 1201))

   
    for finger in range(1, 6):
        # Find all trials corresponding to the current finger
        finger_trials = trial_points[trial_points["finger"] == finger]

        # Initialize a list to store brain responses for all trials of the finger
        erp_data = []

        # Extract and store the ECOG signal for each trial of the current finger within the defined time window.
        for _, trial in finger_trials.iterrows():
            start_index = int(trial["start"]) - 200  
            end_index = int(trial["start"]) + 1000  

            start_index = max(start_index, 0)  
            end_index = min(end_index, len(ecog_data) - 1)  

            erp_trial = ecog_data["ecog"].iloc[start_index:end_index + 1].values
            erp_data.append(erp_trial)

        # Compute the mean brain response for the current finger
        fingers_erp_mean[finger - 1, :] = np.mean(erp_data, axis=0)

    # Plot the graph
    plt.figure(figsize=(10, 6)) 
    colors = ["pink", "purple", "skyblue", "mediumseagreen", "yellow"]  
    for i in range(5):
        plt.plot(fingers_erp_mean[i, :], label=f'Finger {i+1}', color=colors[i])
  
    #Design and display graph
    plt.xlabel('Time (ms)')
    plt.ylabel('Brain Signal (μV)')
    plt.title('AVG ERP FOR FINGER')  
    plt.legend()
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()  

    #Label the matrix and print it
    fingers_labels = [f"Finger {i+1}" for i in range(5)]
    fingers_erp_mean_df = pd.DataFrame(fingers_erp_mean, index=fingers_labels)
    print("Fingers ERP Mean Matrix:")
    print(fingers_erp_mean_df)

    return fingers_erp_mean_df

# Call the function and display the matrix
fingers_erp_mean = calc_mean_erp("events_file_ordered.csv", "brain_data_channel_one.csv")