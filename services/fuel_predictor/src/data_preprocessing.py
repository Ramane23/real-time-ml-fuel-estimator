import pandas as pd
from loguru import logger
import datetime
import warnings
import numpy as np
from tqdm import tqdm
import math

from src.hopsworks_fs import GetFeaturesFromTheStore
from src.config import config 

#Ignore warnings
warnings.filterwarnings('ignore')

#Creating a class to handle data preprocessing
class DataPreprocessing:
    def __init__(self):
        self.columns_to_drop = ['aircraft_icao_code', 'airline_iata_code', 'airline_icao_code', 'flight_level', 'arrival_airport_iata', 
                   'arrival_airport_icao', 'departure_airport_iata', 'departure_airport_icao', 'flight_icao_number', 'flight_number', 'flight_status', 'departure_airport_lat', 'departure_airport_long', 'arrival_airport_lat', 'arrival_airport_long'
                   ]
        self.static_columns = ['aircraft_iata_code','aircraft_mtow_kg','aircraft_malw_kg', 'aircraft_engine_class', 'aircraft_num_engines', 
                  'airline_name', 'arrival_city', 'departure_city', 'flight_id', 'departure_country', 'arrival_country',
                  'route', 'engine_type','isground']

        self.dynamic_columns = ['latitude', 'longitude', 'altitude', 'direction','horizontal_speed','vertical_speed','true_airspeed_ms','bypass_ratio',
                   'cruise_thrust','cruise_sfc','ei_nox_to','ei_nox_co','ei_nox_app','ei_nox_idl','ei_co_to','ei_co_co','ei_co_app','ei_co_idl',
                   'co2_flow','h2o_flow','co_flow','nox_flow','hc_flow', 'soot_flow', 'fuel_flow']
        pass

    #Method to retrieve the flights data from the hopsworks feature store
    def _get_flight_data(self) -> pd.DataFrame:
        
        """a method to retrieve the flights data from the hopsworks feature store

        Returns:
            flights_data (pd.DataFrame): a pandas dataframe containing the flights data
        """
        logger.debug('Retrieving the flights data from the hopsworks feature store...')
        #Instantiate the GetFeaturesFromTheStore class
        get_features_from_the_store = GetFeaturesFromTheStore()
        
        #retrieve the flights data from the hopsworks feature store
        flights_data = get_features_from_the_store.get_features(live_or_historical=config.live_or_historical)
        
        #drop the columns that are not needed
        flights_data.drop(columns=self.columns_to_drop, inplace=True)
        
        # set the index to current_flight_time
        flights_data = flights_data.set_index('current_flight_time')
        
        #cast latitude, longitude to float
        flights_data['latitude'] = flights_data['latitude'].astype(float)
        flights_data['longitude'] = flights_data['longitude'].astype(float)
        
        #Converting the 'current_flight_time' column to a timestamp
        flights_data.index = flights_data.index.map(lambda x: datetime.datetime.fromtimestamp(x))
        
        #create amsk to only keep Air France flights
        mask = flights_data['airline_name'] == 'Air France'
        flights_data = flights_data[mask]
        
        # Remove seconds, focus only on hours and minutes
        flights_data.index = pd.to_datetime(flights_data.index).floor('min')
        
        logger.debug(f"there are {flights_data.shape[0]} rows and {flights_data.shape[1]} columns in the data")
        
        logger.info(flights_data.head(10))
        #breakpoint()
        return flights_data
    
    #Method to preprocess the data
    def floor_and_fill_timestamps(
        self,
        df: pd.DataFrame, 
        flight_id_column: str, 
        keep: str = 'last'
        ) -> pd.DataFrame:
        """
        Floors timestamps to the nearest minute, removes duplicates within the same flight, 
        and fills missing timestamps at 1-minute intervals for each flight.

        Parameters:
        df (pd.DataFrame): Input DataFrame with flight tracking data.
        flight_id_column (str): Column name representing the flight identifier.
        keep (str): Whether to keep 'first' or 'last' duplicate rows within each flight. Default is 'last'.
        
        Returns:
        pd.DataFrame: Processed DataFrame with continuous timestamps.
        """
        # Step 1: Floor the timestamps to the nearest minute
        df.index = pd.to_datetime(df.index).floor('T')
        
        # Step 2: Remove duplicates within the same flight (since data is already sorted)
        df = (
            df
            .groupby(flight_id_column)
            .apply(lambda group: group[~group.index
            .duplicated(keep=keep)])
            .reset_index(level=0, drop=True)
            )
        
        # Step 3: Resample to fill any missing timestamps at 1-minute intervals within each flight group
        df = (
            df
            .groupby(flight_id_column)
            .resample('1T')
            .asfreq()
            .reset_index(level=0, drop=True)
            )
        logger.debug(f"there are {df.shape[0]} rows and {df.shape[1]} columns in the data")
        logger.info(df.head(5))
        return df


    #method to fill missing values
    def fill_missing_values(
        self, 
        flights_data: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Fills missing values in the DataFrame using forward fill.

        Parameters:
        df (pd.DataFrame): Input DataFrame with flight tracking data.

        Returns:
        pd.DataFrame: DataFrame with missing values filled using forward fill.
        """
        #forward fill the static columns
        flights_data[self.static_columns] = flights_data[self.static_columns].ffill()
        #linear interpolate the dynamic columns
        flights_data[self.dynamic_columns] = flights_data[self.dynamic_columns].interpolate(method='linear', limit_direction='both')
        logger.debug(f"there are {flights_data.shape[0]} rows and {flights_data.shape[1]} columns in the data")
        logger.info(flights_data.head(5))
        return flights_data
    
    #method to compute distance and fuel consumption between two points
    # Function to calculate Haversine distance between two points
    def haversine(
        self,
        lat1 : float,
        lon1 : float,
        lat2 : float,
        lon2 : float
        ) -> float:
        
        """this function calculates the Haversine distance between two points

        Returns:
            distance (float): the distance between two points
        """
        R = 6371  # Radius of Earth in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c  # distance in kilometers

    # Function to add distance and fuel consumed columns to the dataframe
    def calculate_distances_and_fuel(
        self,
        df : pd.DataFrame
        )-> pd.DataFrame:
        
        """this function calculates the distance and fuel consumed between two points

        Returns:
            df (pd.DataFrame): the dataframe with the distance and fuel consumed columns
        """
        distances = [0]  # Start with 0 for the first row
        fuel_consumed = [0]  # Start with 0 for the first row

        # Loop through the dataframe
        for i in range(1, len(df)):
            lat1, lon1 = df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude']
            lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
            
            t1, t2 = df.index[i - 1], df.index[i]  # Assuming 'current_flight_time' is the index
            fuel_flow1, fuel_flow2 = df.iloc[i - 1]['fuel_flow'], df.iloc[i]['fuel_flow']
            
            # Only calculate if flight_id is the same for consecutive rows
            if df.iloc[i]['flight_id'] == df.iloc[i - 1]['flight_id']:
                # Calculate distance using Haversine formula
                distance = self.haversine(lat1, lon1, lat2, lon2)
                distances.append(distance)

                # Calculate the time difference in seconds
                time_diff = (t2 - t1).total_seconds()

                # Calculate average fuel flow rate
                avg_fuel_flow = (fuel_flow1 + fuel_flow2) / 2

                # Calculate fuel consumed for this time interval (in kg or lbs depending on your units)
                fuel_consumed_segment = (avg_fuel_flow * time_diff) 
                fuel_consumed.append(fuel_consumed_segment)
            else:
                # If flight_id is different, set distance and fuel consumed to 0 for this segment
                distances.append(0)
                fuel_consumed.append(0)

        # Add new columns to the dataframe
        df['distance_since_last_km'] = distances
        df['fuel_consumed_since_last_kg'] = fuel_consumed

        return df
    
    #method to preprocess the data
    def apply_preprocessing(
        self
        ) -> pd.DataFrame:
        
        """this method applies the preprocessing steps to the flights data
        """
        # Step 1: Retrieve the flights data
        flights_data = self._get_flight_data()
        
        # Step 2: Floor and fill timestamps
        flights_data = self.floor_and_fill_timestamps(flights_data, 'flight_id')
        
        # Step 3: Fill missing values
        flights_data = self.fill_missing_values(flights_data)
        
        # Step 4: Calculate distances and fuel consumed
        flights_data = self.calculate_distances_and_fuel(flights_data)
        
        #groupby route and calculate the total distance and fuel consumed
        route_grouped = flights_data.groupby(['route','current_flight_time']).agg(
            total_distance=('distance_since_last_km', 'sum'),
            total_fuel_consumed=('fuel_consumed_since_last_kg', 'sum')
            ).reset_index(level='route')
        logger.debug(f"there are {route_grouped.shape[0]} rows and {route_grouped.shape[1]} columns in the data")
        logger.info(route_grouped.head(5))
        
        return route_grouped
    
if __name__ == '__main__':
        
    data_preprocessing = DataPreprocessing()
    data_preprocessing.apply_preprocessing()