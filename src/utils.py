import yaml
from omegaconf import OmegaConf

def load_config(path: str = "config.yml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return OmegaConf.create(data)

