# Outlier Detection for the Motion Sense Dataset

This repository contains code that implements the iForestASD and Half Space Tree algorithms for outlier detection on the MotionSense dataset.

## Dataset

The MotionSense dataset consists of sensor data collected from wearable devices. For this project, the stationary status, i.e., standing and sitting, is treated as outliers. Subject 20 dataset is selected, and the original dataset is modified to contain only 1% of stationary status.

- Dataset files:
    - revised_sub20_data_2ndV.csv
    - revised_sub20_cleandata_forStream.csv
    - motion_sense_combined_data
    - revised_sub20_data_Ishu_modification.csv

The dataset is read chunk by chunk, and each chunk contains 1,000 rows of data. The data in each chunk is simulated as streaming data that is read one by one. Half Space Tree, iForestASD, LOF and Heirarichal clustering algorithms are used to detect outliers in the Motion Sense dataset.

The motion sense combined data was created by joining the motion type in chronological order of each subject. The revised_sub20_data_Ishu_modification is modified motion type of subject #20 by manually reducing the stationary movement type (sitting and standing) to just 1% of the data.

## iForestASD Algorithm

The iForestASD algorithm is implemented using modular functions obtained from the [skmultiflow_IForestASD](https://github.com/MariamBARRY/skmultiflow_IForestASD) repository.

### Overview 
The iForestASD algorithm operates on streaming data by leveraging the concept of isolation forests and half space trees. It uses a sliding window approach to process data in chunks. The algorithm dynamically updates the model by fitting it with the reference data and adapts to changing data distributions.

### Pseudo code summary
The core functionality of the iForestASD algorithm is implemented through classes and functions. Details can be found in the skmultiflow_IForestASD repository. Here is a summary of the key components in pseudocode:

- **IsolationForestStream** class:
  - Initializes parameters and maintains the sliding window.
  - Provides methods for partial fitting, updating the model, computing anomaly rates, and making predictions.

- **IsolationTreeEnsemble** class:
  - Fits and maintains an ensemble of isolation trees using a random sample from the data.
  - Supports computing path lengths, anomaly scores, and predicting from anomaly scores.

- **IsolationTree** class:
  - Builds an individual isolation tree by recursively splitting the data until a height limit is reached.

- **HSTree** function:
  - Implements the Half Space Tree algorithm on the streaming data.
  - Fits the model with the reference data and iterates over the streaming chunks.
  - Makes predictions using the Half Space Tree model and partially fits the model with new incoming data.
  - Calculates and prints the results for each chunk.

- **run_comparison()** function:
- Note: The run_comparison() function performs a comparison between the HSTree method and the iForestASD method. It evaluates the performance of these models on a given stream of data using prequential evaluation. The function has the following functionalities:
  - Sets up necessary parameters and imports the required modules.
  - Creates a directory for storing the result files and defines the models to be evaluated. The models used are:
    - HSTree: This model is created using the HalfSpaceTrees class from the skmultiflow.anomaly_detection module. It is configured with parameters such as the number of features in the stream, window size, number of estimators, and anomaly threshold.
    - iForestASD: This model is created using the IsolationForestStream class from the source.iforestasd_scikitmultiflow module. It also takes parameters such as window size, number of estimators, anomaly threshold, and drift threshold.
  - Performs prequential evaluation using the EvaluatePrequential class from the skmultiflow.evaluation.evaluate_prequential module. The evaluator is set up with parameters such as the pretrain size, maximum number of samples, metrics to be computed (e.g., accuracy, F1 score, kappa, running time, model size), batch size, and the path for storing the evaluation results.
  - Runs the evaluation by calling the evaluate() method of the evaluator. The stream of data and the models to be evaluated are passed as arguments. The model names are provided as a list, including "HSTrees" and "iForestASD".
  - Prints the location of the evaluation results.



## Half Space Tree Algorithm

### Pseudo code summary
- **HSTree(buffer, anomaly_threshold, depth, n_estimators, random_state, size_llimit, window_size)** function:
    - Initialize variables and data structures
    - Extract the first reference_start chunks of data from the buffer
    - Create a Half Space Tree model and fit it with the reference data
    - Iterate over each chunk of data in the buffer:
        - Create a new instance of the Half Space Tree model
        - Fit the model with the reference data
        - Iterate over each row in the current chunk of data:
            - Make a prediction using the Half Space Tree model
            - Partially fit the model with the new incoming data and the prediction
            - Append the prediction to the predictions list
        - Add prediction and matched columns to the current chunk of data
        - Calculate and print the results for the current chunk
        - Calculate accuracy for the chunk and append it to the accuracy list
    - Return the accuracy list
 
### LOF algorithm
We use the revised_sub20_data_Ishu_modification.csv file. This is the data of subject 20 whose motions are combined. So this guy has series of motions like walking, jogging, standing, downstairs, upstairs and sitting. We have taken stationary phase (sitting and standing) as outliers and reduced its occurence to just 1% of the data, i.e., 99% of the time series data is either walking, jogging, running, downstairs or upstairs and 1% is sitting and standing.

We read the data chunk by chunk again and can adjust the lookback period. We can either use the previous chunk to predict the next chunk or use all the chunks till now to predict the next chunk (which is arguably slower and may or may not perform better). We use the  LocalOutlierFactor function from scikit learn to predict. 

### Hierarichal clustering algorithm
We again make use of the revised_sub20_data_Ishu_modification.csv file. We first create all the linkages by either using "single", "complete", "ward" or "average" distance between the clusters. Then, we fit an Hierarichal clustering model by using the AgglomerativeClustering function of the scikit learn and use it for all the combinations of the linkages. 

We also change the distance metric and try different connectivity with all the combinations tried and results given.


## Results
- Half Space Tree has a moderate performance on outlier detection for the Motion Sense datset. 
    - Accuracy: 0.6774
    - Macro averaged precision: 0.5028
    - Macro averaged recall: 0.5554
    - Macro averaged f1-score: 0.4175
    - It takes 1.6264016033953918e-06 seconds to process each row of data on average when running in MSiA Jupyter Hub.

- iForestASD takes much longer time to run. When window size is set to be 50 and n_estimator is set to be 30, iForstASD has the following performance:
    - Accuracy: 0.9885
    - Precision: 0.3300
    - Recall: 0.3328
    - F1 score: 0.4419
    - It takes 0.17185152 seconds to process each row of data on average when running in MSiA Jupyter Hub.

- LOF, depending upon the lookback period takes from < 1 second to ~13 seconds (use all the chunks to grow reference to predict the next chunk). Interestingly, its accuracy is highest when we use only the previous chunk. This can be due to autocorrelation where the latest datapoint is most correlated and therefore offers the best insight.
      - Time: 0.42s
      - F1: 0.11813
      - Precision: 0.063151
      - Recall: 0.91265

- Hierarichal clustering offers different results for different combinations; L2 distance with connectivity = 2 and # clusters = 5. I will give the results of the best combination here: Interestingly enough, when we differentiate between all the motion types including sitting and standing (rather than saying that they are one motion type because the person is stationary, we still differentiate between these two), we are able to separate out the sitting motion type with 100% accuracy. This may be due to the fact that a person may be slightly leaning forward or backward when standing and therefore is not really perfectly stationary as assumed before. When he is sitting, he has the support of, say, chair and therefore doesn't lean at all allowing it to be separated. The confusion matrix of the best result is given below:

                  labels        0      1       2       3      4
            type                                         
            dws      3901.0    0.0   267.0  1583.0    1.0
            jog      2208.0    0.0  1389.0  1829.0  102.0
            sit         0.0  157.0     0.0     0.0    0.0
            std         1.0    0.0   174.0     0.0    0.0
            ups      5025.0    0.0   648.0  1036.0   10.0
            wlk     13868.0    0.0  1014.0     2.0   17.0



# Outlier Detection for the Divvy Bike Dataset
This repository contains code that implements the moving average (MA) method for outlier detection on the Divvy Bike dataset.

## Dataset
The Divvy bike dataset is a collection of data related to bike sharing in Chicago area. It includes information such as bike trips, start and end stations, timestamps, and bike availability. The dataset allows for analysis and understanding of bike usage patterns and trends in the Divvy bike sharing system. The data of the first quarter of 2023 is analyzed.

- Dataset files:
    - 202301-divvy-tripdata.csv
    - 202302-divvy-tripdata.csv
    - 202303-divvy-tripdata.csv

## Main Ideas

### Goal
The objective is to determine whether the bike data recorded within a specific one-hour interval on a given date can be classified as an outlier. For instance, we may question if the count of bikes traveling between 10:00 and 11:00 on March 21st, 2023, which is 4, should be considered as an outlier.

### Methodology
To identify if the bike data during a specific one hour interval on a specific date is an outlier or not, bike data should be separately considered based on different pairs of start station and end station. For every single pairs of stations, three dimensions are considered: time, date, and stations.
- Firstly,since there exists rush hours that leads to more bikes during some time than other time, a time series data that contains the number of bikes going from the start station to the end station at a specific one-hour interval for every day in the first quarter of 2023 is generated.
- The second time series generated is the number of bikes going from the start station to the end station for every one-hour time interval, i.e. from 0:00 to 24:00, on a specific date.
- The third data generated is across stations. The number of bikes going from all pairs of stations during a specific one-hour interval on a specific date is found.
- Moving Average is utilized to identify outliers within the generated datasets, allowing for the detection of significant deviations from the expected bike count patterns.

### Time Series Generation c

The first function, bikes_check_out_by_hour, takes three inputs: start_station, end_station, and hour_interval, and returns the number of bikes running from the start_station to the end_station during the specified one-hour interval for each day in the first quarter of 2023.
- **bikes_check_out_by_hour(start_station, end_station, hour)** function:
    - Read the trip data from the first quarter of 2023
    - Filter the data by start and end station
    - Calculate the checkout time by hour
    - Group the data by date and checkout hour to get bike counts
    - Generate a time series with all hours in the first quarter of 2023
    - Fill in the bike counts for checkout hours in the time series
    - Get the bike counts for the specified hour interval
    - Return the hour counts

The second function, bikes_check_out_by_date, takes three inputs: start_station, end_station, and date, and returns the number of bikes going from the start_station to the end_station for each one-hour time interval on the specified date in the first quarter of 2023.
- **bikes_check_out_by_date(start_station, end_station, date)** function:
    - Convert the date to a datetime object
    - Calculate the end date as the next day
    - Filter the data by start and end station and the specified date
    - Calculate the checkout time by hour
    - Group the data by date and checkout hour to get bike counts
    - Generate a time series with all hours in the specified date
    - Join the bike counts with the timestamp dataframe
    - Fill in missing bike counts with 0
    - Rename the column to 'bike count'
    - Return the timestamp and bike counts dataframe  

The third function, bikes_check_out_by_date, takes three inputs: start_station, end_station, and date, and returns the number of bikes going from the start_station to the end_station for each one-hour time interval on the specified date in the first quarter of 2023.
- **find_bikes_between_stations(hour, date)** function:
    - Split the hour range into start hour and end hour
    - Create start time by combining the date and start hour
    - Create end time by combining the date and end hour
    - Filter the data based on the start time and end time
    - Group the data by start and end stations and count the number of bikes
    - Reset the index of the resulting DataFrame
    - Return the DataFrame with start station, end station, and bike counts

### Moving Average Pseudo code summary
The moving_average function utilizes the moving average method to detect outliers in streaming data. It takes the following parameters: window_size (the size of the moving window), outlier_threshold (the threshold to determine if a data point is an outlier), df (the DataFrame containing the data), and target (the data point to be evaluated).

- **moving_average(window_size, outlier_threshold, df, target)** function:
    - Calculate rolling mean using the specified window size
    - Calculate rolling standard deviation using the specified window size
    - If rolling mean or rolling standard deviation is not available:
        - Print "Not enough data to compute Z-score."
    - else:
        - Calculate Z-score for the target data point:
            - Subtract the rolling mean from the target data point and divide by the rolling standard deviation
        - If the Z-score is greater than the outlier threshold:
            - Print "New data point is an outlier!"
        - else:
            - Print "New data point is not an outlier."

## Results
An illustrative example can be found in the DivvyBike-Outlier-Detection.ipynb notebook. In this example, we have chosen the one-hour interval of 2023-03-15 08:00-09:00. The outcomes reveal whether the quantity of bikes traveling between station pairs during this particular one-hour interval is deemed an outlier, utilizing the three dimensions elucidated previously.
