import yaml

def load_yaml(yaml_path: str) -> dict:
    """Load yaml as dictionary.

    Args:
        yaml_path (str): Path to yaml file.

    Returns:
        dict: Parsed yaml.
    """
    with open(yaml_path, mode="r") as f:
        return yaml.safe_load(f)
