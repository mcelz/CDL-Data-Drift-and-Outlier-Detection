from typing import List, Union, Optional, Tuple
import numpy as np
import math

def ERW(X: Union[List[float], np.ndarray],
        n: int,
        base_classifier: object,
        retrain: bool = False,
        num_std: int = 4,
        train_labels: Optional[Union[List[int], np.ndarray]] = None) -> Tuple[List[int], int]:
    """
    Detects change points in a time series using the Ensemble Relevance Window (ERW) algorithm.

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
    c4 = math.sqrt(2/(n-1))*math.factorial(int(n/2) - 1)/math.factorial(int((n-1)/2) - 1)
    
    #create sliding windows
    ref = []
    cur = []
    cur_idx = []
    idx = -1
    lst = []
    m_lst = []
    s_lst = []

    for i in X:
        idx += 1
        # calculate x
        prob = base_classifier.predict_proba([i])
        y_index = np.argmax(prob)
        y = [0]*6
        y[y_index] = 1
        distance = np.sum((np.array(y) - np.array(prob))**2).round(4)
        # ensemble sliding window
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
            #append to current window lists
            m_lst.append(np.mean(cur))
            s_lst.append(np.mean(cur)) 
            #calculate reference window mean
            m_ref = np.mean(ref)
            #determine if drift happens
            if np.mean(m_lst) + num_std*(np.std(m_lst)/sqrt_n)*c4 <= m_ref:
                lst.append(idx-n)
                if retrain:
                    #learn a new base classifier
                    base_classifier.probability=True
                    base_classifier.fit(X[cur_idx], train_labels[cur_idx])
                #initialize window as empty
                ref = []
                cur = []
                cur_idx = []
                m_lst = []
                s_lst = []   
    
    return lst, len(lst)
