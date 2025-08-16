# sentenial-x/apps/ml_pipeline/config.py
import os
import json

class MLPipelineConfig:
    """
    Configuration class for the Sentenial-X ML pipeline.
    """

    def __init__(self,
                 feedback_file="secure_db/feedback.json",
                 model_type="RandomForest",
                 model_params=None,
                 batch_size=32,
                 epochs=10,
                 validation_split=0.2,
                 normalize_method="standard",
                 missing_strategy="mean",
                 log_level="INFO",
                 log_file="logs/ml_pipeline.log",
                 model_output="secure_db/model.pkl",
                 metrics_output="secure_db/metrics.json"):
        
        # Data settings
        self.feedback_file = feedback_file
        self.normalize_method = normalize_method
        self.missing_strategy = missing_strategy

        # Model settings
        self.model_type = model_type
        self.model_params = model_params or {"n_estimators": 100, "max_depth": None, "random_state": 42}

        # Training settings
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

        # Logging settings
        self.log_level = log_level
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Output settings
        self.model_output = model_output
        self.metrics_output = metrics_output
        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_output), exist_ok=True)

    def save(self, path: str):
        """
        Save configuration to a JSON file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
    
    @staticmethod
    def load(path: str):
        """
        Load configuration from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file {path} does not exist")
        with open(path, "r") as f:
            cfg_dict = json.load(f)
        return MLPipelineConfig(**cfg_dict)

# Example usage:
# config = MLPipelineConfig()
# config.save("secure_db/config.json")
# loaded_config = MLPipelineConfig.load("secure_db/config.json")