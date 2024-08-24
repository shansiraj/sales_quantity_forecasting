import yaml
import logging

def get_config(config_path):
    """
    Loads configuration parameters from a YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise