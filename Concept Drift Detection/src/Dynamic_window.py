from typing import List, Union, Optional, Tuple
import numpy as np
import math

def DMRW(X: Union[List[float], np.ndarray],
         base_classifier: object,
         retrain: bool = False,
         num_std: float = 2,
         train_labels: Optional[Union[List[int], np.ndarray]] = None) -> Tuple[List[int], int]:    
    """
    Detects change points in a time series using the Dynamic Moving Range Window (DMRW) algorithm.

    Args:
        X: The input time series data.
        base_classifier: The base classifier used for change detection. It should have a `predict_proba` method.
        retrain: Whether to retrain the base classifier when a change point is detected.
        num_std: The number of standard deviations to consider for change point detection.
        train_labels: The training labels corresponding to the input \
            time series data. Required only if `retrain` is True.

    Returns:
        tuple: A tuple containing the list of change point indices and the total number \
            of change points detected.

    """
    #window size n
    min_window_size = 100
    max_window_size = 400
    cur_window_size = min_window_size
    ref_window_size = min_window_size
    cur_window_size = min_window_size

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

        # move sliding windows
        if len(ref) < cur_window_size:
            ref.append(distance)
        elif len(cur) < cur_window_size:
            cur.append(distance)
            cur_idx.append(idx)

        #dynamically updating the window size
        elif len(lst)>2 and lst[-2]-lst[-1] > 2*cur_window_size:
            cur_window_size = max(min_window_size,cur_window_size-min_window_size)
        elif len(lst)>12 and lst[len(lst)//2]-lst[-1] < cur_window_size:
            cur_window_size = min(max_window_size,cur_window_size+min_window_size)

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
            if m_cur >= m_ref + num_std*(s_ref/math.sqrt(cur_window_size)):
                lst.append(idx-cur_window_size)

                #learn a new base classifier
                if retrain:
                    base_classifier.probability=True
                    base_classifier.fit(X[cur_idx], train_labels[cur_idx])

                #initialize window as empty
                cur_idx = []
                ref = []
                cur = [] 
    return lst, len(lst)
