import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger(__name__)

def load_data(config_path):
    logger.info("Loading California Housing dataset.")
    data = fetch_california_housing(as_frame=True)
    df = data['frame']
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

def preprocess_data(data, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Preprocessing data - splitting into training and testing sets.")
    
    # Extract features and target
    X = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=config['preprocessing']['test_size'], 
                                                        random_state=config['preprocessing']['random_state'])
    
    logger.info(f"Data split complete: {len(X_train)} training samples and {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test
