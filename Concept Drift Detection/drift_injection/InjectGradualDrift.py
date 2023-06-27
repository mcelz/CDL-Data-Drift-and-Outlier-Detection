import random
import pandas as pd
import numpy as np

def inject_gradual_drift(data, target_variable, task_type, drift_type, instances_of_drift, magnitude_of_drift, drift_length):
    df = data.copy(deep=True)

    drift_start_index = int(len(df) / 2)

    section_iterator = int(drift_start_index / instances_of_drift)

    target_variable_index = df.columns.get_loc(target_variable)

    if task_type == "classification":
        number_of_classes = df[target_variable].nunique()

    if ((len(df) - drift_start_index) / drift_length) < instances_of_drift:
        print("Error: either reduce the drift_length or reduce instances_of_drift")
        return

    # gradual drift

    probability_if_random = 1 / number_of_classes

    if magnitude_of_drift == "low":
        drift_amount = 0.5
    elif magnitude_of_drift == "medium":
        drift_amount = 1
    elif magnitude_of_drift == "high":
        drift_amount = 2

    class_list = []
    for i in range(number_of_classes):
        class_list.append(i + 1)

    if task_type == "regression":
        for i in range(instances_of_drift):
            for j in range(drift_length):
                df.iloc[(drift_start_index + (i * section_iterator)) + j, target_variable_index] = df.iloc[(drift_start_index + (i * section_iterator)) + j, target_variable_index] * (1 + ((drift_amount * j) / drift_length))

    if task_type == "classification":
        for i in range(instances_of_drift):
            for j in range(drift_length):
                class1_prob = probability_if_random + ((drift_amount * j) / drift_length * probability_if_random)
                class_prob_list = []
                for k in range(number_of_classes):
                    if k == 0:
                        class_prob_list.append(class1_prob)
                    else:
                        class_prob_list.append((1 - class1_prob) / (number_of_classes - 1))
                df.iloc[(drift_start_index + (i * section_iterator)) + j, target_variable_index] = np.random.choice(class_list, 1, p=class_prob_list)

    return df
