import dagshub.auth
import mlflow
import logging
from sklearn.ensemble import RandomForestRegressor
import yaml
from sklearn.metrics import mean_squared_error
import dagshub
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch repo details from environment variables
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
dagshub.auth.tokens.add_app_token(os.getenv("DAGSHUB_TOKEN"))

# Initialize DagsHub for MLflow integration
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, config_path):
    logger.info("Training RandomForest Regressor model.")
    
    # Load configuration for the model
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = RandomForestRegressor(
        n_estimators=config['model']['n_estimators'], 
        max_depth=config['model']['max_depth'], 
        random_state=config['model']['random_state']
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating the model.")
    
    # Predict and calculate MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Model evaluation complete. Mean Squared Error: {mse:.4f}")
    return mse

def log_to_mlflow(model, mse, config_path):
    logger.info("Logging to MLflow.")
    
    # Log metrics and the model to MLflow
    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "random_forest_regressor")
    
    logger.info("Logged model and metrics to MLflow.")
