from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model, log_to_mlflow
from src.utils.logger import setup_logger
from src.utils.progress import PipelineProgress

def main():
    logger = setup_logger(log_file='pipeline.log')
    
    # Log pipeline start
    logger.info("Starting the MLOps pipeline for Regression Task...")
    
    # Define the total number of steps in the pipeline
    total_steps = 4
    progress = PipelineProgress(total_steps)
    
    try:
        config_path = "src/config.yaml"

        # Step 1: Load the data
        logger.info("Step 1/4: Loading data...")
        progress.update("Loading data")
        data = load_data(config_path)
        
        # Step 2: Preprocess the data
        logger.info("Step 2/4: Preprocessing data...")
        progress.update("Preprocessing data")
        X_train, X_test, y_train, y_test = preprocess_data(data, config_path)
        
        # Step 3: Train the model
        logger.info("Step 3/4: Training the model...")
        progress.update("Training model")
        model = train_model(X_train, y_train, config_path)
        
        # Step 4: Evaluate and log the model
        logger.info("Step 4/4: Evaluating and logging the model...")
        progress.update("Evaluating and logging model")
        mse = evaluate_model(model, X_test, y_test)
        log_to_mlflow(model, mse, config_path)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {str(e)}")
    finally:
        progress.close()

if __name__ == "__main__":
    main()
