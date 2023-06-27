#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skmultiflow.anomaly_detection import HalfSpaceTrees
import glob
from collections import deque
import dask.dataframe as da
import math
import itertools
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings("ignore")

# Data Path
dat1 = pd.read_csv('../datasets/revised_sub20_data_Ishu_modification.csv')

import re
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

all_files = sorted(glob.glob(r"revised_sub20_data_Ishu_modification.csv"), key=numericalSort)
li = []

window_len = 1000
buffer = deque()
for file in all_files:
    window_start = 0
    dfs = pd.read_csv(all_files[0], iterator = True, chunksize = window_len)
    for idx, df in enumerate(dfs):
        ## Use this df chunk
        buffer.append(df)


# Buffer is a deque of pandas dataframe. While its use is like a list, we made it a deque in case in future we need to append at the end or left or need to pop from left or right and we can do it in less time complexity


outlier_buffer = deque()
reference = pd.DataFrame()
reference_end = 3
reference = pd.concat(list(itertools.islice(buffer, 0, reference_end))) # concat dataframes from start to end-1

change_reference = 1
# 0 means don't change the reference on which the model was trained (static)
# 1 means change it and make it equal to the previous chunk
# 2 means grow the reference. So it has all the previous chunks. This may slow down the training but sometimes gives better performance
accuracy_only_outliers, accuracy = [], []

columns_to_use_numerical_all = [i for i in range(4, 16)] # Only use these columns to train
columns_to_use_numerical_better = [5,7,8] # Using parsimonious columns gives better results. These 3 perform better than giving all the columns
columns_to_use = columns_to_use_numerical_better # Using parsimonious columns to get better results

start = time.time()
for i in range(len(buffer)):
    if i < reference_end:
        continue # Start predicting when we are not in the reference
    lof_novelty = LocalOutlierFactor(n_neighbors=10, novelty=True).fit(reference.iloc[:, columns_to_use]) # Train on reference
    prediction_novelty = lof_novelty.predict(buffer[i].iloc[:, columns_to_use]) # Predict for the next chunk
    # Change the anomalies' values to make it consistent with the true values
    prediction_novelty = [1 if i==-1 else 0 for i in prediction_novelty] # Change -1 to 1 and 1 to 0
    
    ones_zeroes_series = pd.Series(prediction_novelty) # convert this 1 and 0 array to pd.Series to later convert to a dataframe
    series_value_counts = pd.Series(prediction_novelty).value_counts(dropna = False) # Get value counts of 0s and 1s
    series_total = series_value_counts.sum() # Basically, total (0s + 1s)
    buffer[i]['pred'] = ones_zeroes_series.values # Prediction column
    buffer[i]['matched'] = np.where(buffer[i]['outlier'] == buffer[i]['pred'], 1, 0) # Matched to the label or not. 1 denotes successfull match and 0 denotes mismatch
    
    # Check the model performance. Commented out to get clean output
    
    # print("**************************************************************")
    # print(f"Results for idx = {i} and rows from {buffer[i].index.min()} to {buffer[i].index.max()} is")
    # print(series_value_counts)
    # print(f'Performance for idx = {i} and rows from {buffer[i].index.min()} to {buffer[i].index.max()} is')
    # print(buffer[i]['matched'].value_counts(dropna = False))
    # print("**************************************************************")
    # print(f'Ground truth for idx = {i} and rows from {buffer[i].index.min()} to {buffer[i].index.max()} is')
    # print(buffer[i]['outlier'].value_counts(dropna = False))
    # display(buffer[i])
    # print(buffer[i].matched.value_counts(dropna = False))
    # print(buffer[i].pred.value_counts(dropna = False))
    
    print("**************************************************************")
    print(f"Results for idx = {i} and rows from {buffer[i].index.min()} to {buffer[i].index.max()} is")
    
    acc = sum(buffer[i]['matched'])/len(buffer[i]['matched']) # Accuracy is matched == 1 / total
    print('Accuracy is ' + str(acc))
    accuracy.append(acc)
    
    # Check if the chunk had outlier to get outlier detection accuracy and append it to the accuracy_only_outliers else append -1
    if len(buffer[i][buffer[i]['outlier'] == 1]) > 0:
        acc_only_outliers = sum(buffer[i][(buffer[i]['matched'] == 1) & (buffer[i]['outlier'] == 1)]['matched']) / len(buffer[i][buffer[i]['outlier'] == 1])
        print('Outlier accuracy is ' + str(acc_only_outliers))
        accuracy_only_outliers.append(acc_only_outliers)
    else:
        print("No outliers in this buffer chunk")
        accuracy_only_outliers.append(-1)
        
    if change_reference == 1:
        reference = buffer[i]
    elif change_reference == 2:
        reference = pd.concat([reference, buffer[i]])
    elif change_reference == 0:
        pass
    else:
        raise("Unimplemented error for change_reference flag meaning")
        

end = time.time()
total_time = end - start

print(f"Total accuracy = {np.mean(accuracy)}")
print(f"Only outlier detection accuracy = {np.mean([i for i in accuracy_only_outliers if i != -1])}") # Accuracy of only outliers

print(f"Total time taken is {total_time} seconds")

truth = []
prediction = []
for i in range(reference_end,len(buffer)):
    truth += list(buffer[i].outlier)
    prediction += list(buffer[i].pred)

print(f"Confusion matrix")
print(pd.DataFrame(confusion_matrix(truth,prediction)))

print('F1 Score is {:.5}'.format(f1_score(truth,prediction)))

print('Recall is {:.5}'.format(recall_score(truth,prediction)))
print('Precision is {:.5}'.format(precision_score(truth,prediction)))


