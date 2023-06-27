import time
import argparse
import csv
import sys
import yaml
import pandas as pd
import src.window_optimization as window_opt
import src.data_drift_detection as data_drift


if __name__ == "__main__":

    ## Obtaining the configurations
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config", default="config/default-config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            sys.exit(1)
        except FileNotFoundError:
            sys.exit(1)

    ## Reference data
    ref_total = pd.read_csv('/Users/kiranjyothisheena/Documents/Kiran_Files/CDL_Practicum/Datasets/Motion_Sense/A_DeviceMotion_data/jog_9/sub_2.csv')
    ref_total.drop("Unnamed: 0",axis=1,inplace=True)
    ref_total = ref_total['attitude.roll']

    ## Obtaining the optimum window size
    max_window = window_opt.return_optimum_window_size(ref_total)

    para_config = config['run_parameters']
    chunk_size = para_config.get('chunk_size',1) ## Chunk size with which data is fed in
    ## Maximum consecutive data points after which drift is alarmed
    max_threshold = para_config.get('max_threshold', 40)
    curr_threshold_check = 0 ## Counter for checking consecutive points

    result_pd = pd.DataFrame()
    return_dataframe = False ## Boolean to decide if results need to be returned (For debug only)

    curr_buff = pd.Series() ## Dataframe to hold the test values

    ## Helper function to simulate streaming using csv values
    def stream_csv_data(filename):
        """
        Summary: Function to simulate streaming data

        Input:
        filename: Name of csv file
        """
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Read the header row
            for row in reader:
                yield dict(zip(headers, row))
                time.sleep(0.5)  # Simulate a delay of 0.5 second between each row

    ## Test data
    folder_location = '/Users/kiranjyothisheena/Documents/Kiran_Files/CDL_Practicum/Datasets/Motion_Sense/Test_datasets/'
    csv_filename = folder_location + 'Subject_2_11_distributed_lambda=2000_size=1000_50.csv'

    curr_t = max_window
    for data_row in stream_csv_data(csv_filename):
        curr_value = float(data_row['attitude.roll']) ## Reading the new value

        ## Checking if curr buffer size is less than window
        ## If yes, we fill the buffer else we test for drift
        if len(curr_buff) < max_window:
            curr_buff = pd.concat([curr_buff,pd.Series(curr_value)])
        else:
            if return_dataframe:
                curr_result,curr_res_dataframe = \
                data_drift.return_drift_indicator(curr_buff,ref_total,return_dataframe)    
            else:
                curr_result = \
                    data_drift.return_drift_indicator(curr_buff,ref_total,return_dataframe)

            if curr_result==1:
                curr_threshold_check+=1 ## Increasing the counter if we detect drift
            else:
                curr_threshold_check = 0

            if curr_threshold_check > max_threshold:
                print('Drift Detected')
                break

            ## Updating the buffer
            curr_buff = curr_buff[chunk_size:]
            curr_buff = pd.concat([curr_buff,pd.Series(curr_value)])

            if return_dataframe:
                curr_res_dataframe['index'] = 't=' + str(curr_t)
                curr_t += 1
                result_pd = pd.concat([result_pd,curr_res_dataframe])
