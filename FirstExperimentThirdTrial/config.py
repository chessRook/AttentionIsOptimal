import yaml

from typing import Dict
from pathlib import Path


class Config:
    def __init__(self):
        self.config_path = Path('./config.yaml')
        self.config = self.config_loader()

    def config_loader(self):
        with open(self.config_path, "r") as yaml_stream:
            yaml_content: Dict[str, str] = yaml.safe_load(yaml_stream)
        return yaml_content

    def __getitem__(self, attribute):
        if attribute not in self.config:
            raise AttributeError('No such attribute')

        value = self.config[attribute]
        return value


# Debug Code

cfg = Config()
print(cfg.config)
