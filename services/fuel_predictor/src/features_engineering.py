import pandas as pd
from loguru import logger
import warnings
from typing import Tuple
from openap import prop, FuelFlow, Emission
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_preprocessing import DataPreprocessing

#ignore warnings
warnings.filterwarnings('ignore')

class FeaturesEngineering(BaseEstimator, TransformerMixin):
    """
    Class to engineer features for the contrail predictor using OpenAP aircraft performance model
    """
    #Instantiate the DataPreprocessing class
    data_preprocessing = DataPreprocessing()
    
    def __init__(self):
        
      
      pass
    
    #method to generate temporal features

    def generate_temporal_features(
        self, 
        rolling_window=5
        ) -> pd.DataFrame:
        """ this method generates temporal features

        Args:
            rolling_window (int, optional): The rolling window for the features. Defaults to 5.

        Returns:
            pd.DataFrame: The dataframe with the temporal features
        """
        
        df = self.data_preprocessing.apply_preprocessing()

        # Ensure 'current_flight_time' (the index) is a datetime type
        df.index = pd.to_datetime(df.index)

        # 1. Hour of the day
        df['hour_of_day'] = df.index.hour

        # 2. Minute of the hour
        df['minute_of_hour'] = df.index.minute

        # 3. Day of the week
        df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6

        # 4. Day of the month
        df['day_of_month'] = df.index.day

        # 5. Month
        df['month'] = df.index.month

        # Adding Rolling Averages for total_distance and total_fuel_consumed
        df['rolling_avg_distance'] = df.groupby('route')['total_distance'].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
        df['rolling_avg_fuel'] = df.groupby('route')['total_fuel_consumed'].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())

        # Adding Rolling Sum (useful for cumulative tracking over short periods)
        df['rolling_sum_distance'] = df.groupby('route')['total_distance'].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).sum())
        df['rolling_sum_fuel'] = df.groupby('route')['total_fuel_consumed'].transform(lambda x: x.rolling(window=rolling_window, min_periods=1).sum())

        # Features for distance flown and fuel consumed during the last hour(s)
        # Distance and fuel for last 1 hour
        df['distance_last_hour'] = df.groupby('route')['total_distance'].transform(lambda x: x.rolling('60T').sum())
        df['fuel_last_hour'] = df.groupby('route')['total_fuel_consumed'].transform(lambda x: x.rolling('60T').sum())

        # Distance and fuel for last 2 hours
        df['distance_last_2_hours'] = df.groupby('route')['total_distance'].transform(lambda x: x.rolling('120T').sum())
        df['fuel_last_2_hours'] = df.groupby('route')['total_fuel_consumed'].transform(lambda x: x.rolling('120T').sum())

        # Distance and fuel for last 3 hours
        df['distance_last_3_hours'] = df.groupby('route')['total_distance'].transform(lambda x: x.rolling('180T').sum())
        df['fuel_last_3_hours'] = df.groupby('route')['total_fuel_consumed'].transform(lambda x: x.rolling('180T').sum())

            
        logger.info('Temporal features generated successfully')
        logger.debug(f"There are {df.shape[0]} rows and {df.shape[1]} columns in the dataset")
        logger.info(df.head())

        return df
            

    #method to separate the features and target variable
    def separate_features_target(
        self, 
        df: pd.DataFrame
        ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Method to separate the features and target variable
        """
        X = df.drop(columns=['total_fuel_consumed'])
        y = df['total_fuel_consumed']

        return X, y

    #method to engineer features
    def apply_features_engineering(
        self, 
        #df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Method to apply feature engineering
        """
        # Generate temporal features
        df = self.generate_temporal_features()
        #separate features and target variable
        X, y = self.separate_features_target(df)
        
        logger.info('Features engineered successfully')
        logger.debug(f"There are {X.shape[0]} rows and {X.shape[1]} columns in the features dataset")
        logger.debug(f"There are {y.shape[0]} rows in the target variable dataset")
        logger.info(X.head())
        logger.info(y.head())

        return X, y
        
    
if __name__ == '__main__':
    
    #Instantiate the FeaturesEngineering class
    features_engineering = FeaturesEngineering()
    #apply feature engineering
    features_engineering.apply_features_engineering()
    
    
    


