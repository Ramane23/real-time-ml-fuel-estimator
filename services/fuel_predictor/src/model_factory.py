from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor  # XGBoost Regressor
from lightgbm import LGBMRegressor  # LightGBM Regressor
from sklearn.linear_model import LinearRegression  # Linear Regressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from comet_ml import Experiment
from loguru import logger
import pandas as pd
from typing import Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the existing FeaturesEngineering class
from src.features_engineering import FeaturesEngineering

from sklearn.preprocessing import LabelEncoder
from loguru import logger

class CategoricalFeatureEncoder:

    def __init__(self, categorical_columns):
        """ Initialize an empty dictionary to store encoders for each column. """
        self.label_encoders = {}
        self.categorical_columns = categorical_columns

    def fit(self, X):
        """
        Fit label encoders for the specified categorical columns.

        Args:
            X (pd.DataFrame): The input data.
            categorical_columns (list): The list of columns to encode.
        """
        for col in self.categorical_columns:
            logger.info(f"Fitting encoder for column: {col}")
            le = LabelEncoder()
            # Fit on the entire column to learn all categories
            self.label_encoders[col] = le.fit(X[col])

    def transform(self, X):
        """
        Transform the categorical columns using the fitted label encoders.

        Args:
            X (pd.DataFrame): The input data.
        """
        X_transformed = X.copy()

        for col, le in self.label_encoders.items():
            logger.info(f"Transforming column: {col}")
            
            # Handle unknown categories by assigning a new label (use -1 for unknown)
            X_transformed[col] = X_transformed[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        return X_transformed

    def fit_transform(self, X):
        """
        Fit and transform the data for the specified categorical columns.

        Args:
            X (pd.DataFrame): The input data.
            categorical_columns (list): The list of columns to encode.
        """
        self.fit(X)
        X_encoded = self.transform(X)
        #breakpoint()
        return X_encoded

    def inverse_transform(self, X):
        """
        Inverse transform the encoded columns back to their original labels.

        Args:
            X (pd.DataFrame): The input data with encoded columns.
        """
        X_inversed = X.copy()

        for col, le in self.label_encoders.items():
            logger.info(f"Inverse transforming column: {col}")
            X_inversed[col] = le.inverse_transform(X_inversed[col])
        
        return X_inversed


# Main Model Class for Linear Regression
class LinearRegressionPipelineModel:
    def __init__(
        self,
        experiment: Experiment
        ):
        self.features_engineering = FeaturesEngineering()
        self.experiment = experiment
    
    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamp_split: str,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing based on a given timestamp.

        Args:
            X (pd.DataFrame): The input feature dataframe.
            y (pd.Series): The target values.
            timestamp_split (str): The timestamp to split the data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
        """
        df = pd.concat([X, y], axis=1)
        cutoff_timestamp = pd.Timestamp(timestamp_split)
        train_data = df[df.index < cutoff_timestamp]
        test_data = df[df.index >= cutoff_timestamp]
        X_train, y_train = train_data.drop(columns=[y.name]), train_data[y.name]
        X_test, y_test = test_data.drop(columns=[y.name]), test_data[y.name]
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        description: Optional[str] = 'linear Regressor Model Evaluation'
        ) -> dict:
        """
        Evaluates the regression model using MSE, RMSE, MAE, and R-squared.

        Args:
            model: The trained pipeline model.
            test_data (pd.DataFrame): Test dataframe containing features and target.
            description (str): Description of the evaluation.

        Returns:
            dict: Evaluation metrics.
        """
        logger.info(f'**** {description} ****')
        
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and log regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r2
        }

        logger.info(f"Metrics: {metrics}")
        
        return metrics

    def fit_linear_regression_pipeline(
        self,
        timestamp_split: str,
        X_transformed: pd.DataFrame, 
        y_transformed : pd.Series
        ) -> LinearRegression:
        """
        Prepare the data using feature engineering and fit a Linear Regression model.
        """
        
        logger.info("Applying feature engineering transformations...")

        # Apply feature engineering to get transformed X and y
        #X_transformed , y_transformed = self.features_engineering.apply_features_engineering()
        
        # Log a dataset hash to track the data
        self.experiment.log_dataset_hash(X_transformed)
        #log the list of features used in the model
        self.experiment.log_parameter('features', X_transformed.columns.tolist())
        
        # Encode categorical features
        categorical_columns = ['route']
        categorical_encoder = CategoricalFeatureEncoder(categorical_columns=categorical_columns)
        X_transformed = categorical_encoder.fit_transform(X_transformed)
        
        # Step 2: Split the data into train and test sets based on the timestamp
        logger.info(f"Splitting the data into train and test sets at {timestamp_split}...")
        X_train, X_test, y_train, y_test  = self.split_train_test(X_transformed, y_transformed, timestamp_split)
        logger.debug(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        logger.debug(f"Train target shape: {y_train.shape}, Test target shape: {y_test.shape}")

        # Log the number of rows in train and test datasets
        self.experiment.log_metric('n_rows_train', X_train.shape[0])
        self.experiment.log_metric('n_rows_test', X_test.shape[0])
        self.experiment.log_metric('n_features', X_train.shape[1])
        self.experiment.log_metric('n_target', y_train.shape[0])

        # Scale features for better performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        #breakpoint()

        logger.info(f"Fitting Linear Regression model on {X_train_scaled.shape[0]} samples...") 

        # Fit the Linear Regression model
        linear_regressor = LinearRegression()
        linear_regressor.fit(X_train_scaled, y_train)
        
        # Evaluate the model on the test set
        logger.info("Evaluating the Linear Regression model...")
        linear_metrics = self.evaluate_model(
            model=linear_regressor, 
            X_test= X_test_scaled,
            y_test = y_test
            )
        linear_regressor_rmse = linear_metrics['RMSE']
        self.experiment.log_metrics(linear_metrics)

        logger.info(f"Linear Regression model fitting complete with performance: {linear_metrics}")
        
        return linear_regressor,linear_regressor_rmse
        
        

# Main Model Class for XGBoost Regressor
class XGBoostPipelineModel:
    def __init__(
        self,
        experiment: Experiment
        ):
        self.features_engineering = FeaturesEngineering()
        self.experiment = experiment
        
    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamp_split: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing based on a given timestamp.

        Args:
            X (pd.DataFrame): The input feature dataframe.
            y (pd.Series): The target values.
            timestamp_split (str): The timestamp to split the data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
        """
        df = pd.concat([X, y], axis=1)
        cutoff_timestamp = pd.Timestamp(timestamp_split)
        train_data = df[df.index < cutoff_timestamp]
        test_data = df[df.index >= cutoff_timestamp]
        X_train, y_train = train_data.drop(columns=[y.name]), train_data[y.name]
        X_test, y_test = test_data.drop(columns=[y.name]), test_data[y.name]
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        description: Optional[str] = 'XGBoost Model Evaluation'
        ) -> dict:
        """
        Evaluates the regression model using MSE, RMSE, MAE, and R-squared.

        Args:
            model: The trained pipeline model.
            test_data (pd.DataFrame): Test dataframe containing features and target.
            description (str): Description of the evaluation.

        Returns:
            dict: Evaluation metrics.
        """
        logger.info(f'**** {description} ****')

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and log regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r2
        }

        logger.info(f"Metrics: {metrics}")
        
        return metrics
    
    def fit_xgboost_pipeline(
        self,
        timestamp_split: str,
        X_transformed: pd.DataFrame, 
        y_transformed : pd.Series
        ) -> XGBRegressor:
        """
        Prepare the data using feature engineering and fit the XGBoost regressor.
        """
         
        logger.info("Applying feature engineering transformations...")

        # Apply feature engineering to get transformed X and y
        #X_transformed , y_transformed = self.features_engineering.apply_features_engineering()
        
        # Log a dataset hash to track the data
        self.experiment.log_dataset_hash(X_transformed)
        #log the list of features used in the model
        self.experiment.log_parameter('features', X_transformed.columns.tolist())
        
        # Encode categorical features
        categorical_columns = ['route']
        categorical_encoder = CategoricalFeatureEncoder(categorical_columns=categorical_columns)
        X_transformed = categorical_encoder.fit_transform(X_transformed)
        
        # Step 2: Split the data into train and test sets based on the timestamp
        logger.info(f"Splitting the data into train and test sets at {timestamp_split}...")
        X_train, X_test, y_train, y_test  = self.split_train_test(X_transformed, y_transformed, timestamp_split)
        logger.debug(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        logger.debug(f"Train target shape: {y_train.shape}, Test target shape: {y_test.shape}")

        # Log the number of rows in train and test datasets
        self.experiment.log_metric('n_rows_train', X_train.shape[0])
        self.experiment.log_metric('n_rows_test', X_test.shape[0])
        self.experiment.log_metric('n_features', X_train.shape[1])
        self.experiment.log_metric('n_target', y_train.shape[0])
        
        # Scale features for better performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        logger.info(f"Fitting XGBoost model on {X_train_scaled.shape[0]} samples...") 

        # Fit the Linear Regression model
        xgb_regressor = XGBRegressor()
        xgb_regressor.fit(X_train_scaled, y_train)
        
        # Evaluate the model on the test set
        logger.info("Evaluating the XGBoost model...")
        xgb_metrics = self.evaluate_model(
            model=xgb_regressor, 
            X_test= X_test_scaled,
            y_test = y_test
            )
        xgb_rmse = xgb_metrics['RMSE']
        self.experiment.log_metrics(xgb_metrics)

        logger.info(f"XGBoost model fitting complete with performance: {xgb_metrics}")
        
        return xgb_regressor,xgb_rmse


# Main Model Class for LightGBM Regressor
class LightGBMPipelineModel:
    def __init__(
        self,
        experiment: Experiment
        ):
        self.features_engineering = FeaturesEngineering()
        self.experiment = experiment

    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        timestamp_split: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing based on a given timestamp.

        Args:
            X (pd.DataFrame): The input feature dataframe.
            y (pd.Series): The target values.
            timestamp_split (str): The timestamp to split the data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
        """
        df = pd.concat([X, y], axis=1)
        cutoff_timestamp = pd.Timestamp(timestamp_split)
        train_data = df[df.index < cutoff_timestamp]
        test_data = df[df.index >= cutoff_timestamp]
        X_train, y_train = train_data.drop(columns=[y.name]), train_data[y.name]
        X_test, y_test = test_data.drop(columns=[y.name]), test_data[y.name]
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        description: Optional[str] = 'LightGBM Model Evaluation'
        ) -> dict:
        """
        Evaluates the regression model using MSE, RMSE, MAE, and R-squared.

        Args:
            model: The trained pipeline model.
            test_data (pd.DataFrame): Test dataframe containing features and target.
            description (str): Description of the evaluation.

        Returns:
            dict: Evaluation metrics.
        """
        logger.info(f'**** {description} ****')

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and log regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r2
        }

        logger.info(f"Metrics: {metrics}")
        
        return metrics
    
    def fit_lightgbm_pipeline(
        self,
        timestamp_split: str,
        X_transformed: pd.DataFrame, 
        y_transformed : pd.Series 
        ) -> XGBRegressor:
        """
        Prepare the data using feature engineering and fit the XGBoost regressor.
        """
        logger.info("Applying feature engineering transformations...")

        # Apply feature engineering to get transformed X and y
        #X_transformed , y_transformed = self.features_engineering.apply_features_engineering()
        
        # Log a dataset hash to track the data
        self.experiment.log_dataset_hash(X_transformed)
        #log the list of features used in the model
        self.experiment.log_parameter('features', X_transformed.columns.tolist())
        
        # Encode categorical features
        categorical_columns = ['route']
        categorical_encoder = CategoricalFeatureEncoder(categorical_columns=categorical_columns)
        X_transformed = categorical_encoder.fit_transform(X_transformed)
        
        # Step 2: Split the data into train and test sets based on the timestamp
        logger.info(f"Splitting the data into train and test sets at {timestamp_split}...")
        X_train, X_test, y_train, y_test  = self.split_train_test(X_transformed, y_transformed, timestamp_split)
        logger.debug(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        logger.debug(f"Train target shape: {y_train.shape}, Test target shape: {y_test.shape}")

        # Log the number of rows in train and test datasets
        self.experiment.log_metric('n_rows_train', X_train.shape[0])
        self.experiment.log_metric('n_rows_test', X_test.shape[0])
        self.experiment.log_metric('n_features', X_train.shape[1])
        self.experiment.log_metric('n_target', y_train.shape[0])

        # Scale features for better performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        logger.info(f"Fitting Linear Regression model on {X_train_scaled.shape[0]} samples...") 

        # Fit the LightGBM model
        lgb_regressor = LGBMRegressor()
        lgb_regressor.fit(X_train_scaled, y_train)
        
        # Evaluate the model on the test set
        logger.info("Evaluating the Linear Regression model...")
        lgb_metrics = self.evaluate_model(
            model=lgb_regressor, 
            X_test= X_test_scaled,
            y_test = y_test
            )
        lgb_rmse = lgb_metrics['RMSE']
        self.experiment.log_metrics(lgb_metrics)

        logger.info(f"LightGBM model fitting complete with performance metrics: {lgb_metrics}")
        
        return lgb_regressor,lgb_rmse   