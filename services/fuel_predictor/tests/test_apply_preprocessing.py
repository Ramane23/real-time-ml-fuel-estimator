import pytest
from src.data_preprocessing import DataPreprocessing
import pandas as pd

def test_apply_preprocessing():
    # Instantiate the DataPreprocessing class
    data_preprocessing = DataPreprocessing()
    # Call the apply_preprocessing method
    df = data_preprocessing.apply_preprocessing()
    # Check that the method returns a DataFrame
    assert isinstance(df, pd.DataFrame)
    #check that current_flight_time is the index
    assert df.index.name == 'current_flight_time'
    # Check that the DataFrame has the expected columns
    expected_columns = [
        'total_distance',
        'total_fuel_consumed',
        'route'
    ]
    assert all(col in df.columns for col in expected_columns)
    
    #check the types of the columns
    assert df['total_distance'].dtype == 'float64'
    assert df['total_fuel_consumed'].dtype == 'float64'
    assert df['route'].dtype == 'object'
    