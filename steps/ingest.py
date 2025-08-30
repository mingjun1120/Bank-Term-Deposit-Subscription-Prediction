import pandas as pd
import yaml

class Ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open("config.yml", "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        train_data_path = self.config['data']['train_path']
        test_data_path = self.config['data']['test_path']
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        return train_data, test_data
    
dvc remote modify --system myremote exclude_environment_credential true
dvc remote modify --system myremote exclude_visual_studio_code_credential true
dvc remote modify --system myremote exclude_shared_token_cache_credential true
dvc remote modify --system myremote exclude_managed_identity_credential true