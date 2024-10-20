import pytest
from src.features_engineering import FeaturesEngineering
import pandas as pd

def test_apply_preprocessing():
    # Instantiate the DataPreprocessing class
    features_engineering =  FeaturesEngineering()
    # Call the apply_preprocessing method
    X_transformed, y_transformed = features_engineering.apply_features_engineering()
    # Check that X_transformed is a DataFrame
    assert isinstance(X_transformed, pd.DataFrame)
    #check that y_transformed is a Series
    assert isinstance(y_transformed, pd.Series)
    #check that current_flight_time is the index
    assert X_transformed.index.name == 'current_flight_time'
    # Check that the DataFrame has the expected columns
    expected_columns = ['route', 'total_distance', 'hour_of_day', 'minute_of_hour',
       'day_of_week', 'day_of_month', 'month', 'rolling_avg_distance',
       'rolling_avg_fuel', 'rolling_sum_distance', 'rolling_sum_fuel',
       'distance_last_hour', 'fuel_last_hour', 'distance_last_2_hours',
       'fuel_last_2_hours', 'distance_last_3_hours', 'fuel_last_3_hours']
    
    assert all(col in X_transformed.columns for col in expected_columns)
    
    #check the types of the columns
    assert X_transformed.dtypes['route'] == 'object'
    assert X_transformed.dtypes['total_distance'] == 'float64'
    assert X_transformed.dtypes['hour_of_day'] == 'int64'
    assert X_transformed.dtypes['minute_of_hour'] == 'int64'
    assert X_transformed.dtypes['day_of_week'] == 'int64'
    assert X_transformed.dtypes['day_of_month'] == 'int64'
    assert X_transformed.dtypes['month'] == 'int64'
    assert X_transformed.dtypes['rolling_avg_distance'] == 'float64'
    assert X_transformed.dtypes['rolling_avg_fuel'] == 'float64'
    assert X_transformed.dtypes['rolling_sum_distance'] == 'float64'
    assert X_transformed.dtypes['rolling_sum_fuel'] == 'float64'
    assert X_transformed.dtypes['distance_last_hour'] == 'float64'
    assert X_transformed.dtypes['fuel_last_hour'] == 'float64'
    assert X_transformed.dtypes['distance_last_2_hours'] == 'float64'
    assert X_transformed.dtypes['fuel_last_2_hours'] == 'float64'
    assert X_transformed.dtypes['distance_last_3_hours'] == 'float64'
    assert X_transformed.dtypes['fuel_last_3_hours'] == 'float64'
    
    