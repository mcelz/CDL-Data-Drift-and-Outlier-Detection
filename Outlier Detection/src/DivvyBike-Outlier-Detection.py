#!/usr/bin/env python3

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats


def bikes_check_out_by_hour(start_station, end_station, hour):
    """
    Calculate the number of bikes going between start_station and end_station at a specific hour interval for every day in quarter 1.

    Args:
        start_station (str): The name of the start station.
        end_station (str): The name of the end station.
        hour (str): The specified hour interval in the format 'HH:MM-HH:MM'.

    Returns:
        pd.Series: A pandas Series containing the bike counts for the specified hour interval.

    """
    df1 = pd.read_csv("datasets/202301-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df2 = pd.read_csv("datasets/202302-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df3 = pd.read_csv("datasets/202303-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df = pd.concat([df1, df2, df3])
    df.dropna(inplace=True)
    
    # Filter the data by start and end station
    df = df[(df["start_station_name"] == start_station) & (df["end_station_name"] == end_station)]

    # Get the checkout time by hour
    df["checkout_hour"] = df["started_at"].dt.floor("H")

    # Group by date and checkout hour
    bike_counts = df.groupby(["checkout_hour"])["ride_id"].count()

    # Generate time series data with all hours in March
    idx = pd.date_range(start="2023-01-01", end="2023-03-31", freq="H")
    ts_data = pd.Series(0, index=idx)

    # Fill in bike counts for checkout hours in March
    for checkout_hour, bike_count in bike_counts.iteritems():
        ts_data.loc[checkout_hour] = bike_count

    # Get the bike counts for the specified hour
    hour_counts = ts_data.between_time(hour, hour)

    return hour_counts


def bikes_check_out_by_date(start_station, end_station, date):
    """
    Calculate the number of bikes checked out between start_station and end_station for each hour of the specified date.

    Args:
        start_station (str): The name of the start station.
        end_station (str): The name of the end station.
        date (str): The date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the timestamp and bike counts for each hour of the specified date.

    """
    df1 = pd.read_csv("datasets/202301-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df2 = pd.read_csv("datasets/202302-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df3 = pd.read_csv("datasets/202303-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df = pd.concat([df1, df2, df3])
    df.dropna(inplace=True)
    
    start_date = pd.to_datetime(date)
    end_date = start_date + pd.Timedelta(days=1)
    df = df[(df["start_station_name"] == start_station) & 
            (df["end_station_name"] == end_station) & 
            (df["started_at"] >= start_date) & 
            (df["started_at"] < end_date)]

    # Get the checkout time by hour
    df["checkout_hour"] = df["started_at"].dt.floor("H")

    # Group by date and checkout hour
    bike_counts = df.groupby(["checkout_hour"])["ride_id"].count()

    # Generate time series data with all hours in the specified date
    start_datetime = pd.Timestamp(date)
    end_datetime = start_datetime + pd.Timedelta(days=1)
    idx = pd.date_range(start=start_datetime, end=end_datetime, freq="H")
    ts_data = pd.DataFrame({"timestamp": idx})

    # Join the bike counts with the timestamp dataframe
    ts_data = pd.merge(ts_data, bike_counts, how="left", left_on="timestamp", right_index=True)
    ts_data["ride_id"] = ts_data["ride_id"].fillna(0)
    ts_data.rename(columns={'ride_id': 'bike count'}, inplace=True)

    return ts_data


def find_bikes_between_stations(hour, date):
    """
    Find the start station and end station pairs that have bikes checked out between the specified hour and date.

    Args:
        hour (str): The hour range in the format 'HH:MM-HH:MM'.
        date (str): The date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the start station, end station, and the number of bikes for each pair.

    """
    df1 = pd.read_csv("datasets/202301-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df2 = pd.read_csv("datasets/202302-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df3 = pd.read_csv("datasets/202303-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df = pd.concat([df1, df2, df3])
    df.dropna(inplace=True)
    
    # Filter the data for the specified hour and date
    start_hour, end_hour = hour.split("-")
    start_time = pd.Timestamp(date + " " + start_hour)
    end_time = pd.Timestamp(date + " " + end_hour)
    

    filtered_df = df[(df["started_at"] >= start_time) & (df["started_at"] < end_time)]

    # Group by start and end stations and count the number of bikes
    bike_counts = filtered_df.groupby(["start_station_name", "end_station_name"]).size().reset_index(name="bike_count")
    bike_counts.reset_index(drop=True, inplace=True)
    
    return bike_counts


# a helper function that concat hour and date to a timestamp

def convertion(hour: str, date: str) -> pd.Timestamp:
    """
    Convert the hour and date strings to a pandas Timestamp object.

    Args:
        hour (str): The hour range in the format 'HH:MM-HH:MM'.
        date (str): The date in the format 'YYYY-MM-DD'.

    Returns:
        pd.Timestamp: The desired Timestamp object with the specified date and hour.

    """
    # Convert date and hour to pandas Timestamp objects
    date = pd.Timestamp(date)
    hour_start, hour_end = hour.split("-")
    hour_start = pd.Timestamp(hour_start)
    hour_end = pd.Timestamp(hour_end)

    # Combine date and hour to get the desired Timestamp
    desired_ts = pd.Timestamp(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=hour_start.hour,
        minute=hour_start.minute,
    )
    return desired_ts



def moving_average(window_size, outlier_threshold, df, target):
    '''
    Using moving average method to detect outliers for streaming data
    '''
    rolling_mean = df['bike count'].rolling(window_size).mean()
    rolling_std = df['bike count'].rolling(window_size).std()

    # Set the ANSI escape code for red text
    red_color_code = '\033[91m'
    # Reset the color back to default
    reset_color_code = '\033[0m'
        
    if rolling_mean.empty or rolling_std.empty:
        print("Not enough data to compute Z-score.")
    else:
        if (target - rolling_mean.iloc[-1]) / rolling_std.iloc[-1] > outlier_threshold:
            print(f"{red_color_code}New data point is an outlier!{reset_color_code}")
        else:
            print("New data point is not an outlier.")



def outlier_across_stations(df, outlier_threshold, target):
    '''
    Using mean and std method to detect outliers for static data for different stations
    '''
    mean = df['bike_count'].mean()
    std = df['bike_count'].std()


    if (target - mean) / std > outlier_threshold:
        # Set the ANSI escape code for red text
        red_color_code = '\033[91m'
        # Reset the color back to default
        reset_color_code = '\033[0m'
        # Print the statement in red
        print(f"{red_color_code}New data point is an outlier!{reset_color_code}")
    else:
        print("New data point is not an outlier.")



def main(hour, date, df):
    '''
    The main function that detects outliers for bike data during a specified one-hour interval on a specified date.
    The function loops every pair of stations at the time and detect outliers.
    '''
    hour_date = convertion(hour,date)
    
    # Sort the routes by trip count in descending order
    bikes_data_station = find_bikes_between_stations(hour, date)
    print("********Detecting if there are outliers on " + str(hour_date) + "********")
    for i in range(bikes_data_station.shape[0]):
        start_station = bikes_data_station.iloc[i,0]
        end_station = bikes_data_station.iloc[i,1]
        
        print("From "+ start_station + " to " + end_station)
    
        bikes_data_hour = pd.DataFrame(bikes_check_out_by_hour(start_station, end_station, hour)).reset_index().rename(columns={'index': 'timestamp', 0: 'bike count'})
        bikes_data_date = bikes_check_out_by_date(start_station, end_station, date)
    
        
        #detect outliers at the same time across days
        target = int(bikes_data_hour[bikes_data_hour.timestamp == hour_date]['bike count'])
        filtered_df = bikes_data_hour.loc[bikes_data_hour['timestamp'] < hour_date]
        print("Comparing with data at the same time across days: ")
        moving_average(10, 3, filtered_df, target)
    
        #detect outliers at the same day across hours
        target = int(bikes_data_hour[bikes_data_hour.timestamp == hour_date]['bike count'])
        filtered_df = bikes_data_hour.loc[bikes_data_hour['timestamp'] < hour_date]
        print("Comparing with data at the same day across hours: ")
        moving_average(10,3, filtered_df, target)
    
        #detect outliers across stations
        target = bikes_data_station[(bikes_data_station.start_station_name == start_station) 
                                    & (bikes_data_station.end_station_name == end_station)]['bike_count']
        target_location = target.index[0]
        rest_data = bikes_data_station.drop(target_location)
        print("Comparing with data at the same day across stations: ")
        outlier_across_stations(rest_data,3,int(target))



if __name__ == "__main__":
    #running the functions
    df1 = pd.read_csv("datasets/202301-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df2 = pd.read_csv("datasets/202302-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df3 = pd.read_csv("datasets/202303-divvy-tripdata.csv", parse_dates=["started_at", "ended_at"])
    df = pd.concat([df1, df2, df3])

    #here is an example
    hour = "08:00-09:00"
    date = "2023-03-15"
    main(hour, date, df)


