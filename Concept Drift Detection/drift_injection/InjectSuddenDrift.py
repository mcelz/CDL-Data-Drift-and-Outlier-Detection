import random
import pandas as pd

def inject_sudden_drift(data, target_variable, task_type, drift_type, instances_of_drift, magnitude_of_drift, drift_length):
    df = data.copy(deep=True)

    drift_start_index = int(len(df) / 2)

    # Checking
    if ((len(df) - drift_start_index) / drift_length) < instances_of_drift:
        print("Error: either reduce the drift_length or reduce instances_of_drift")
        return

    section_iterator = int(drift_start_index / instances_of_drift)

    target_variable_index = df.columns.get_loc(target_variable)

    if task_type == "classification":
        number_of_classes = df[target_variable].nunique()

        for i in range(instances_of_drift):
            for j in range(drift_length):
                df.iloc[(drift_start_index + (i * section_iterator)) + j, target_variable_index] = random.randrange(number_of_classes)

    elif task_type == "regression":
        if magnitude_of_drift == "low":
            drift_amount = 0.5
        elif magnitude_of_drift == "medium":
            drift_amount = 1
        elif magnitude_of_drift == "high":
            drift_amount = 2

        for i in range(instances_of_drift):
            for j in range(drift_length):
                df.iloc[(drift_start_index + (i * section_iterator)) + j, target_variable_index] = df.iloc[(drift_start_index + (i * section_iterator)) + j, target_variable_index] * (1 + drift_amount)

    return df

