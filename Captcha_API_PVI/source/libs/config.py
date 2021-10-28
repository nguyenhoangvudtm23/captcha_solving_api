import yaml


def get_config():
    with open('config/configs.yaml', 'rb') as f:
        cfg = yaml.safe_load(f)
    return cfg
