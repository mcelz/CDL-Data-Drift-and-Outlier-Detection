import pandas as pd
import InjectSuddenDrift as igd


data = pd.read_csv("MotionSenseData.csv")  # Your dataset
target_variable = "target"  # Name of the target variable column
task_type = "classification"  # or "regression"
drift_type = "sudden"  # The type of drift (not used in the function)
instances_of_drift = 3  # Number of drift instances to inject
magnitude_of_drift = "high"  # Magnitude of drift ("low", "medium", or "high")
drift_length = 200  # Length of each drift instance

drifted_data_sudden = igd.inject_gradual_drift(data, target_variable, task_type, drift_type, instances_of_drift, magnitude_of_drift, drift_length)


drifted_data_sudden.to_csv("Sudden30high200.csv")