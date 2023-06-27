import numpy as np
import math
from typing import List, Union, Optional, Tuple

def MRW(X: Union[List[float], np.ndarray],
        n: int,
        base_classifier: object,
        retrain: bool = False,
        num_std: int = 2,
        train_labels: Optional[Union[List[int], np.ndarray]] = None) -> Tuple[List[int], int]:
    """
    Detects change points in a time series using the Moving Range Window (MRW) algorithm.

    Args:
        X: The input time series data.
        n: The window size.
        base_classifier: The base classifier used for change detection. It should have a `predict_proba` method.
        retrain: Whether to retrain the base classifier when a change point is detected. 
        num_std: The number of standard deviations to consider for change point detection.
        train_labels: The training labels corresponding to the input time series data. 

    Returns:
        A tuple containing the list of change point indices and the total number of change points detected.
    """
    #window size n
    sqrt_n = math.sqrt(n)

    #create sliding windows
    ref = []
    cur = []
    cur_idx = []
    idx = -1
    lst = []
    for i in X:
        idx += 1
        # calculate x
        prob = base_classifier.predict_proba([i])
        y_index = np.argmax(prob)
        y = [0]*6
        y[y_index] = 1
        distance = np.sum((np.array(y) - np.array(prob))**2).round(4)
        # move sliding window
        if len(ref) < n:
            ref.append(distance)
        elif len(cur) < n:
            cur.append(distance)
            cur_idx.append(idx)
        else:
            #move reference window by 1
            ref.pop(0)
            ref.append(cur[0])
            #move current window by 1
            cur.pop(0)
            cur.append(distance)
            cur_idx.pop(0)
            cur_idx.append(idx)
            #append to current window 
            m_cur = np.mean(cur)
            m_ref = np.mean(ref)
            s_ref = np.std(ref)
            if m_cur >= m_ref + num_std*(s_ref/sqrt_n):
                lst.append(idx-n)
                #learn a new base classifier
                if retrain:
                    base_classifier.probability=True
                    base_classifier.fit(X[cur_idx], train_labels[cur_idx])
                #initialize window as empty
                cur_idx = []
                ref = []
                cur = [] 
    return lst, len(lst)