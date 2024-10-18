import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger

from src.features_engineering import FeaturesEngineering


class BaselineModel:
    
    #Instantiate the FeaturesEngineering class
    features_engineering = FeaturesEngineering()
    
    def __init__(self, X_transformed: pd.DataFrame, y_transformed: pd.Series):
        """
        Initialize the BaselineModel class.
        
        Args:
            features_engineering: An instance of the FeaturesEngineering class to apply feature transformations.
        """
        
        logger.info("Initializing the BaselineModel class...")

        # Step 1: Apply feature engineering and get transformed X and y
        logger.info("Applying feature engineering to get transformed features and target...")
        self.X, self.y = X_transformed, y_transformed

    def naive_mean_predictor(self):
        """Make naive predictions using the mean of the training target variable."""
        logger.info("Making naive predictions using the mean of the training data...")
        
        # Step 2: Calculate the mean of the training target values
        mean_value = self.y.mean()
        logger.info(f"Mean value of training target: {mean_value}")

        # Step 3: Predict the mean value for every sample in the test set
        self.y_pred = np.full_like(self.y, fill_value=mean_value)
        logger.info("Naive predictions complete.")

    def evaluate_model(self):
        """Evaluate the naive model performance."""
        logger.info("Evaluating naive model performance...")

        # Step 4: Make naive predictions
        self.naive_mean_predictor()

        # Step 5: Evaluate Naive Model
        metrics = self.calculate_metrics(self.y, self.y_pred)

        # Display the results
        print("\n=== Naive Model Evaluation Results ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return metrics

    @staticmethod
    def calculate_metrics(true_values, predicted_values):
        """Helper function to calculate regression metrics."""
        try:
            mse = mean_squared_error(true_values, predicted_values)
            metrics = {
                'Mean Squared Error (MSE)': mse,
                'Root Mean Squared Error (RMSE)': np.sqrt(mse),
                'Mean Absolute Error (MAE)': mean_absolute_error(true_values, predicted_values),
                'R-squared (R2)': r2_score(true_values, predicted_values)
            }
        except ValueError as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics = {'MSE': 0, 'RMSE': 0, 'MAE': 0, 'R2': 0}
        
        return metrics
    
# Example usage:
if __name__ == "__main__":
    
    X_transformed, y_transformed = BaselineModel.features_engineering.apply_features_engineering()
    
    # Initialize the baseline model
    baseline_model = BaselineModel(X_transformed=X_transformed, y_transformed=y_transformed)

    # Evaluate the naive model
    baseline_model.evaluate_model()
