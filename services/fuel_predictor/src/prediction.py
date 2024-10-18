import pickle
from pydantic import BaseModel
from typing import List, Union, ClassVar
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from loguru import logger
from comet_ml.api import API
from sklearn.preprocessing import StandardScaler

# Import your project's specific modules
from src.data_preprocessing import DataPreprocessing
from src.features_engineering import FeaturesEngineering
from src.model_factory import CategoricalFeatureEncoder
from src.config import config
from src.utils import get_model_name

logger.debug('Running the prediction module...')
logger.debug(f'Config: {config.model_dump()}')

# Instantiate the DataPreprocessing class
data_preprocessing = DataPreprocessing()
#Instantiate the FeaturesEngineering class
features_engineering = FeaturesEngineering()

#define a model type to represet the union of the two models
#Model = Union[XGBClassifier, LGBMClassifier]

#define a pydantic model to represent the output of the predictor class
class FuelPredictorOutput(BaseModel):
    """
    A Pydantic model to represent the output of the Predictor class for contrail prediction.
    """
    route: str
    fuel_consumption: float
    prediction_time: str

    def to_dict(self):
        return {
            'route': self.route,
            'fuel_consumption': self.fuel_consumption,
            'prediction_time': self.prediction_time,
        }

class Predictor:
    
    def __init__(
        self,
        local_model_path: str,
        feature_group_name: str,
        feature_group_version: int,
        feature_view_name: str,
        feature_view_version: int,
        experiment: str,
        experiment_key: str,
    ):
        self.model = self._load_model_pickle(local_model_path)
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version
        self.feature_view_name = feature_view_name
        self.feature_view_version = feature_view_version
        self.experiment = experiment
        self.experiment_key = experiment_key
        
        
    #the class method is used to create a new instance of the class by just passing the model type and status 
    #and the rest of the attributes are fetched from the model registry, i.e we don't have to pass them as arguments
    #during the instantiation of the class
    @classmethod 
    def from_model_registry(
        cls, 
        model_type: str, 
        status: str,
        ) -> 'Predictor':
        """
        Fetches the model artifact from the model registry, and all the relevant
        metadata needed to make predictions, then returns a Predictor object.

        Args:
            - flight_id: The flight_id for which we want to make predictions.
            - status: The status of the model we want to fetch, e.g., "production".

        Returns:
            - Predictor: An instance of the Predictor class with the model artifact and metadata.
        """

        comet_api = API(api_key=config.comet_ml_api_key)

        # Step 1: Download the model artifact from the model registry
        model = comet_api.get_model(
            workspace=config.comet_ml_workspace,
            model_name=get_model_name(model_type)
        )
        model_versions = model.find_versions(status=status)
        model_version = sorted(model_versions, reverse=True)[0]
        
        # download the model artifact for this `model_version`
        model.download(version=model_version, output_folder='./')
        local_model_path = './best_fuel_estimator.pkl'

        # Step 2: Fetch the relevant metadata from the model registry
        experiment_key = model.get_details(version=model_version)['experimentKey']
        experiment = comet_api.get_experiment_by_key(experiment_key)
        feature_group_name = experiment.get_parameters_summary('feature_group_name')['valueCurrent']
        feature_group_version = int(experiment.get_parameters_summary('feature_group_version')['valueCurrent'])
        feature_view_name = experiment.get_parameters_summary('feature_view_name')['valueCurrent']
        feature_view_version = int(experiment.get_parameters_summary('feature_view_version')['valueCurrent'])

        # Step 3: Return a Predictor object with the model artifact and metadata
        return cls(
            local_model_path=local_model_path,
            feature_group_name=feature_group_name,
            feature_group_version=feature_group_version,
            feature_view_name=feature_view_name,
            feature_view_version=feature_view_version,
            experiment=experiment,
            experiment_key=experiment_key,
        )

    def predict_fuel_consumption(self, route: str) -> FuelPredictorOutput:
        """
        Generates fuel consumption prediction using the model in `self.model`
        and the latest data in the feature store for a given route.

        Args:
            - route (str): The flight route for which to generate a prediction (e.g., "Paris - Marseille").

        Returns:
            - FuelPredictorOutput: A Pydantic model containing the route, prediction (True/False), and prediction time.
        """
        
        #else:
        logger.debug(f"Estimating the fuel consumption on Air france flight route {route}")

        # Step 1: Apply initial preprocessing steps and feature engineering
        logger.debug('Fetching flights data from the Hopsworks online feature store and preprocessing...')
        X_transformed, y_transformed = features_engineering.apply_features_engineering()

        # Step 2: Filter the preprocessed data for the given route
        logger.debug(f'Filtering data for route: {route}')
        route_data = X_transformed[X_transformed['route'] == route]

        # Step 3: Check if the route data is empty and return False for contrail formation
        if route_data.empty:
            logger.warning(f"There are currently no flights tracked on the inputed route: {route}, consequently fuel consumption can not be assessed.")
            return FuelPredictorOutput(
                route=route,
                fuel_consumption=0,
                prediction_time=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
        if X_transformed.empty:
            logger.warning(f"No data available for the inputed route: {route}, consequently fuel consumption can not be assessed.")
            return FuelPredictorOutput(
                route=route,
                fuel_consumption=0,
                prediction_time=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            )
        else:
            logger.debug(f'Feature engineering steps applied successfully to the data for route: {route}.')
            # Encode categorical features
            categorical_columns = ['route']
            categorical_encoder = CategoricalFeatureEncoder(categorical_columns=categorical_columns)
            route_data = categorical_encoder.fit_transform(route_data)

            # Scale features for better performance
            scaler = StandardScaler()
            route_data_scaled = scaler.fit_transform(route_data)
                
            # Step 5: Run the model prediction for the given route
            #the fuel consumption is predicted using the model by calling the predict method
            #and summing the predicted values
            logger.debug('Running inference on the filtered route data...')
            fuel_consumption = self.model.predict(route_data_scaled).sum()

            # Step 6: Capture the prediction time
            prediction_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

            # Step 7: Return the structured output
            return FuelPredictorOutput(
                route=route,
                fuel_consumption=fuel_consumption,
                prediction_time=prediction_time,
            )
        
    def _load_model_pickle(
        self, 
        local_model_path: str):
        """ this function loads a model from a pickle file 

        Args:
            local_model_path (str): the path to the model pickle file

        Returns:
            model: the loaded model
        """
        with open(local_model_path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':

    predictor = Predictor.from_model_registry(
        model_type='Linear Regression',
        status='production',
    )
    prediction : FuelPredictorOutput = predictor.predict_fuel_consumption('Abidjan - Paris')
    logger.debug(f'Prediction: {prediction.to_dict()}')