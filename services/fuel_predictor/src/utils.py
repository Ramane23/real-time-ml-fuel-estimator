def get_model_name(model_type: str) -> str:
    """
    Returns the model registry name for the given model type and target column.

    Args:
        - target_column (str): The target column of the model.
        - model_type (str): The type of the model (e.g., 'xgboost', 'lightgbm').

    Returns:
        - str: The model registry name based on the model type and target column.
    """
    return f"{model_type}_fuel_estimator"