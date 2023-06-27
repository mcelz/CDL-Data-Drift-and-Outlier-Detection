# Data Drift Detection

The data drift detection modules are developed to identify drift in the data by comparing the distribution of the current data series with the reference data series used for training the model. The occurrence of drift is assumed when a statistical difference is observed between the reference and test series. To achieve this, three statistical tests, namely PSI, JS divergence, and KL divergence, are employed to assess the difference in distribution between the two data series. If the combined results from all three tests exceed a certain threshold, the current data point (timestamp) is marked as a drift point. Subsequently, if the number of consecutive drift points surpasses a user-defined threshold, it indicates drift in the data, and retraining is recommended.

The data drift algorithms make use of a sliding window to compare distributions. Given a window size N, the data distribution of consecutive N data points in the test data and the overall distribution of the reference data are compared. When the next data point (or the next chunk of data) is received, the window slides i.e the first point is removed and the latest point is added. Given the significance of the window size in detecting drift, determining an optimal value becomes crucial. To address this, a Bayesian optimization technique is implemented to identify the optimum window size based on the reference data prior to implementating drift detection.

## Pseudo code for overall algorithm

```
Input reference data: ref
Find optimum window size: N using ref 
Start streaming data
Initialize buffer with the first N data point
Initialize counter=0
Check if drift occurs at current point
if drift occurs
    counter++
else
    counter = 0

if counter > user defined threshold
    report drift and exit
else
    Add current data point to buffer
    Remove data point in the beginning
continue
```

The pseudo code and explanation for individual sections are given below

## Window size optimization

### Pseudo code for optimization procedure

```
Input reference data: ref
set drift size = 0.9
Add drift in the reference
    Displace the data points at certain continuous intervals by norm(2,1) * standard deviation * drift size
    Interval between each drift window is exponential distributed
    Size of each drift window is normally distributed
    Find indicators for drift and non-drift points
Run Bayesian optimization
    Consider N as current window size
    Run data drift detection for synthetic drift data
    Find accuracy with current window size using indicators
    Iterate till accuracy is optimized
```

### Functions under src.window_optimization

`return_drift` :- Compares the reference and test series and returns a statistical measure on the difference in drift. The 
tests are conducted using functions present in the `evidently` package under `evidently.calculations.stattests`. Based on the statistical values and pre determined threshold, it is predicted whether drift has occured or not.

`add_new_drift` :- Helper function to add synthetic drift into the data. A parent series, start and end index are passed to the function. The function then shifts the values in the series from start to end by a factor of norm(2,1) * std(data)* drift_size,where drift_size is set by default to 0.9.

`create_test_set` :- Function responsible for generating the test dataset used to evaluate accuracy. It introduces drift into the data by calling the `add_new_drift` function at different sections of the data. The interval between each drift window is exponenitally distributed and the size of each drift window is normally distributed. Thus values are sampled from these distributions and then the start and stop index are altered and passed to `add_new_drift`. Additionally, binary indicators denoting whether drift has occurred or not at each point in the reference are also created. These indicators are used to calculate the accuracy of the algorithm using the current window size.

`check_drift_accuracy` :- Driver function for running the optimization. It takes the reference series as input and generates a test series by incorporating synthetic drift using the `create_test_set` function. This synthetic series is then tested against multiple window sizes and the window size with the best accuracy is determined. A sub function `check_drift_accuracy` is defined within it, which given a window size conducts the streaming data drift detection on the synthetic drift data and returns the accuracy using the binary indicators. This function is then fed into a Bayesian optimizer under the package `BayesianOptimization`. The Bayesian optimizer takes the optimization function and the range of values to be searched as input. The search domain, in this case, consists of window sizes ranging from 100 to 10000, which is divided into five sections. The optimizer is executed in multiple threads to efficiently explore the search space. Finally, the optimum window size is determined by selecting the window size with the maximum accuracy across all sections.

## Data Drift Detection

### Pseudo code for data drift detection

```
Get drift statistics between buffer and reference at current time point
Compare statistics with user defined threshold for each test
For each test
    if stats value > threshold
        return indicator = 1
    else 
        return indicator = 0
If 0.4*PSI_test_indicator + 0.4*JS_test + 0.2*KL_test > overall threshold
    return drift = 1
```

### Functions under src.data_drift_detection

`return_drift` :- Compares the reference and test series and returns a statistical measure on the difference in drift. The 
tests are conducted using functions present in the `evidently` package under `evidently.calculations.stattests`. Based on the statistical values and pre determined threshold, it is predicted whether drift has occured or not.

`return_drift_indicator` :- Calls `return_drift` and obtains the statistical test values. These values are stored in a dataframe. The function compares these values against predefined thresholds and creates additional columns within the dataframe to denote drift indicators. A weighted average of these indicators is then compared with an overall threshold to create a final drift indicator. The function returns this indicator and, if required, the dataframe containing the statistical test values and drift indicators.

## Run implementation

Prior to run, the following packages with the corresponding version numbers must be installed in the python environment

```
bayesian-optimization==1.4.3
evidently==0.3.0
multiprocess==0.70.14
PyYAML==6.0
```
The following configurations must be set in the yaml file config/default-config.yaml

`chunk_size` :- The number of data points added to the buffer at each step

`max_threshold` :- Maximum consecutive positive drift points after which drift is reported

An example driver code for the implementation of data drift detection is added as main.py. The reference and test series can be changed for custom test cases.

## Future Directions

1. Incorporate with streaming infrastucture and extend it using spark code.

2. Develop a similar algorithm to handle categorical variables.

3. Incorporate larger ensemble of tests.

## Reference

1. Evidently documentation :- https://github.com/evidentlyai/evidently

2. Bayesian Optimization documentation :- https://github.com/bayesian-optimization/BayesianOptimization
