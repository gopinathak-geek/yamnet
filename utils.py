# utils.py

import json

class Config:
    def __init__(self, config_dict):
        self.sr = config_dict['SR']
        self.chunk_duration = config_dict['CHUNK_DURATION']
        self.model_path = config_dict['MODEL_PATH']
        self.cry_confirmation_count = config_dict['CRY_CONFIRMATION_COUNT']
        self.cry_indices = config_dict['CRY_INDICES']


def load_config(config_path='config.json'):
    """Load configuration from a JSON file and return Config object."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Config(config_dict)
