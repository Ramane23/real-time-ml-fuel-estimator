import pandas as pd
from loguru import logger
from typing import Tuple, Optional
from comet_ml import Experiment
from sklearn.preprocessing import StandardScaler
from src.model_factory import XGBoostPipelineModel, LightGBMPipelineModel, LinearRegressionPipelineModel
from src.data_preprocessing import DataPreprocessing
from src.features_engineering import FeaturesEngineering
from src.baseline_model import BaselineModel
from src.model_factory import CategoricalFeatureEncoder
from src.config import config

# Display the content of the config file
logger.debug('Training service is starting...')
logger.debug(f'Config: {config.model_dump()}')

# Instantiate the FeaturesEngineering class
features_engineering = FeaturesEngineering()

def train(
    feature_group_name: str,
    feature_group_version: int,
    feature_view_name: str,
    feature_view_version: int,
    timestamp_split: str,
    comet_ml_api_key: str,
    comet_ml_project_name: str,
    comet_ml_workspace: str,
):
    """
    This function trains the regression models using XGBoost, LightGBM, and Linear Regression,
    logs the results to CometML, and evaluates model performance.

    Args:
        feature_group_name (str): The name of the feature group.
        feature_group_version (int): The version of the feature group.
        feature_view_name (str): The name of the feature view.
        feature_view_version (int): The version of the feature view.
        timestamp_split (str): Timestamp for splitting the data into train and test sets.
        comet_ml_api_key (str): API key for CometML.
        comet_ml_project_name (str): Project name for CometML.
        comet_ml_workspace (str): Workspace for CometML.

    Returns:
        None.
    """
    # Create an experiment to log metadata to CometML
    experiment = Experiment(
        api_key=comet_ml_api_key,
        project_name=comet_ml_project_name,
        workspace=comet_ml_workspace,
    )
    
    X_transformed, y_transformed = features_engineering.apply_features_engineering()
    
    # Log feature group and feature view names and versions
    experiment.log_parameter('feature_group_name', feature_group_name)
    experiment.log_parameter('feature_view_name', feature_view_name)
    experiment.log_parameter('feature_group_version', feature_group_version)
    experiment.log_parameter('feature_view_version', feature_view_version)
    
    # Step 3: Build and evaluate the baseline model
    logger.info("Building and evaluating the baseline model...")
    baseline_model = BaselineModel(X_transformed=X_transformed, y_transformed=y_transformed)
    baseline_metrics = baseline_model.evaluate_model()
    baseline_rmse = baseline_metrics['Root Mean Squared Error (RMSE)']
    experiment.log_metrics(baseline_metrics)

    # Step 4: Train Linear Regression model using the custom pipeline
    logger.info("Training Linear Regression model using the custom pipeline...")
    linear_pipeline_model = LinearRegressionPipelineModel(experiment)
    linear_regressor, linear_regressor_rmse = linear_pipeline_model.fit_linear_regression_pipeline(
        timestamp_split = timestamp_split,
        X_transformed = X_transformed,
        y_transformed = y_transformed
        )
    
    # Step 5: Train XGBoost model using the custom pipeline
    logger.info("Training XGBoost model using the custom pipeline...")
    xgb_pipeline_model = XGBoostPipelineModel(experiment)
    xgb_regressor, xgb_rmse = xgb_pipeline_model.fit_xgboost_pipeline(
        timestamp_split = timestamp_split,
        X_transformed = X_transformed,
        y_transformed = y_transformed
        )

    # Step 6: Train LightGBM model using the custom pipeline
    logger.info("Training LightGBM model using the custom pipeline...")
    lgbm_pipeline_model = LightGBMPipelineModel(experiment )
    lgbm_regressor, lgbm_rmse = lgbm_pipeline_model.fit_lightgbm_pipeline(
        timestamp_split = timestamp_split,
        X_transformed = X_transformed,
        y_transformed = y_transformed
        )
   
    # Step 7: Save the best model based on RMSE
    min_rmse = min(linear_regressor_rmse, xgb_rmse, lgbm_rmse)
    
    if min_rmse == linear_regressor_rmse and min_rmse <= baseline_rmse:
        best_model = linear_regressor
        best_model_name = 'Linear Regression_fuel_estimator'
        logger.info("Best model: Linear Regression")
        save_model(best_model, './best_fuel_estimator.pkl')
        logger.info(f"Best model saved to './best_fuel_estimator.pkl'")
        experiment.log_model(name=best_model_name, file_or_folder='./best_fuel_estimator.pkl')
        #push the best model to the model registry
        experiment.register_model(model_name=best_model_name)
        
    elif min_rmse == xgb_rmse and min_rmse <= baseline_rmse:
        best_model = xgb_regressor
        best_model_name = 'XGBoostRegressor_fuel_estimator'
        logger.info("Best model: XGBoost")
        save_model(best_model, './best_fuel_estimator.pkl')
        logger.info(f"Best model saved to './best_fuel_estimator.pkl'")
        experiment.log_model(name=best_model_name, file_or_folder='./best_fuel_estimator.pkl')
        #push the best model to the model registry
        experiment.register_model(model_name=best_model_name)
    else:
        best_model = lgbm_regressor
        best_model_name = 'LGBMRegressor_fuel_estimator'
        logger.info("Best model: LightGBM")
        save_model(best_model, './best_fuel_estimator.pkl')
        logger.info(f"Best model saved to './best_fuel_estimator.pkl'")
        experiment.log_model(name=best_model_name, file_or_folder='./best_fuel_estimator.pkl')
        #push the best model to the model registry
        experiment.register_model(model_name=best_model_name)
        
    
        
def save_model(model, filename: str):
    """
    Save the model as a pickle file.

    Args:
        model: The model to save.
        filename (str): Path to save the model file.
    """
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    logger.info(f"Model saved to {filename}")

if __name__ == '__main__':
    # Run the training pipeline
    train(
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        feature_view_name=config.feature_view_name,
        feature_view_version=config.feature_view_version,
        timestamp_split='2024-10-14 12:00:00',
        comet_ml_api_key=config.comet_ml_api_key,
        comet_ml_project_name=config.comet_ml_project_name,
        comet_ml_workspace=config.comet_ml_workspace,
    )
