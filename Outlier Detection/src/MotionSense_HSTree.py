#!/usr/bin/env python
# coding: utf-8



import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skmultiflow.data.file_stream import FileStream
from skmultiflow.anomaly_detection import HalfSpaceTrees
from source.iforestasd_scikitmultiflow import IsolationForestStream
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
import glob
from collections import deque
import dask.dataframe as da
import itertools
import datetime
import time
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, classification_report
import warnings
warnings.filterwarnings("ignore")




def read_data(file_path, window_len = 1000):
    """
    Reads data from files in chunks and returns a buffer containing the data.

    Parameters:
        file_path (str): The file path or pattern to match files.
        window_len (int): The number of rows in each chunk (default = 1000).

    Returns:
        deque: A deque containing the data chunks read from the files.
    """
    # Read data chunk by chunk. Each chunk has 1000 rows
    numbers = re.compile(r'(\d+)')

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    all_files = sorted(glob.glob(file_path), key=numericalSort)
    li = []

    buffer = deque()
    for file in all_files:
        window_start = 0
        dfs = pd.read_csv(all_files[0], iterator = True, chunksize = window_len)
        for idx, df in enumerate(dfs):
            buffer.append(df)
    
    return buffer
    




def HSTree(buffer, anomaly_threshold = 0.5, depth = 15, n_estimators = 25, random_state = 1, size_llimit = 50, window_size = 250):
    """
    Build and partial fit a Half Space Tree model to detect outliers.

    Parameters:
        buffer (deque): A deque containing the buffered streaming data.
        anomaly_threshold (float): The threshold for classifying outliers (default = 0.5).
        depth (int): The maximum depth of the Half Space Tree (default = 15).
        n_estimators (int): The number of estimators to include in the ensemble (default = 25).
        random_state (int): The random seed for reproducibility (default = 1).
        size_llimit (int): The size limit for leaf nodes in the Half Space Tree (default = 50).
        window_size (int): The size of the reference window for the initial training (default = 250).

    Returns:
        tuple: A tuple containing two lists:
            - accuracy (list): The average prediction accuracy for each chunk of data.
            - times (list): The average time used for making predictions and partial fitting the model for each row.

    """
    # The first three chunks of data is used as reference and they are treated as normal data.
    outlier_buffer = deque()
    reference = pd.DataFrame()
    reference_start = 3
    reference = pd.concat(list(itertools.islice(buffer, 0, reference_start)))
    
    # Initialize a Half Space Tree model and specify hyperparameter values
    HST_model = HalfSpaceTrees(random_state = 1, anomaly_threshold= 0.8)
    # Fit the Half Space Tree model with the reference data
    HST_model.fit(reference.iloc[:, 4:16].values, np.array(reference.outlier))
    
    # Simulate data as streaming data and make predictions using the Half Space Tree model.
    # Record the average prediction accuracy for each chunk of data
    accuracy = []
    # Record the average time used for making prediction for each row and partially fit the model.
    times = []
    for i in range(len(buffer)):
        if i < reference_start:
            continue
        # Initialize a Half Space Tree model and specify hyperparameter values
        HST_model = HalfSpaceTrees(random_state = 1, anomaly_threshold= 0.8)
        # Fit the Half Space Tree model with the reference data
        HST_model.fit(reference.iloc[:, 4:16].values, np.array(reference.outlier))

        predict = []
        for j in range(len(buffer[i])):
                start = time.time()
                # Make prediction
                pred = HST_model.predict([list(buffer[i].iloc[j,4:16].values)])
                if (pred[0] == 0):
                    # Partically fit the Half Space with the new incoming data
                    HST_model.partial_fit(np.array([buffer[i].iloc[j,4:16].values]), [pred[0]])
                times.append(time.time() - start)
                predict.append(pred[0])

        buffer[i]['pred'] = predict
        buffer[i]['matched'] = np.where(buffer[i]['outlier'] == buffer[i]['pred'], 1, 0)

        print(f"Results for idx = {i} and rows from {buffer[i].index.min()} to {buffer[i].index.max()} is")
        acc = sum(buffer[i]['matched'])/len(buffer[i]['matched'])
        print('Accuracy is ' + str(acc))
        accuracy.append(acc)
    
    return accuracy, times




def main():
    buffer = read_data("datasets/revised_sub20_data_2ndV.csv", window_len = 1000)
    accuracy, times = HSTree(buffer, anomaly_threshold = 0.5, depth = 15, n_estimators = 25, random_state = 1, size_llimit = 50, window_size = 250)
    # Combine all chunks of data
    combined_dat = buffer[3]
    for i in range(4,len(buffer)):
        combined_dat = pd.concat([combined_dat, buffer[i]])
    # Prediction Evaluation
    # Confusion Matrix
    print("*****confusion matrix*****")
    print(pd.DataFrame(confusion_matrix(list(combined_dat['outlier']),list(combined_dat['pred']))))

    true_labels = list(combined_dat['outlier']) 
    predicted_labels = list(combined_dat['pred'])

    # Calculate precision, recall, and F1 score
    print("******classification report*******")
    report = classification_report(true_labels, predicted_labels, digits=4)
    print(report)

    print("******averge time used to predict and partially fit the model for each row******")
    print(str(np.mean(np.array(times))) + " seconds running in MSiA Jupyter Hub")
    
    # There are 1000 rows in each chunk
    print("******averge time used to predict and partially fit the model for each row******")
    print(str(np.mean(np.array(times))/1000) + " seconds running in MSiA Jupyter Hub")




if __name__ == "__main__":
    main()






