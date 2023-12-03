from rabetorch.util.io_util import load_yaml


class ConfigObject:
    """Config object to manege configure."""
    def __init__(self, _dcit: dict) -> None:
        """Initialization of ConfigObject.

        Args:
            _dcit (dict): Dictionary config.
        """
        for key, val in _dcit.items():
            if isinstance(val, dict):
                val = ConfigObject(val)
            self.__dict__[key] = val

    def update(self, update_cfg: 'ConfigObject') -> None:
        """Update base config by new config.

        Args:
            update_cfg (ConfigObject): New config to update.
        """
        for key, val in update_cfg.__dict__.items():
            if hasattr(self, key) and isinstance(getattr(self, key), ConfigObject) and isinstance(val, ConfigObject):
                getattr(self, key).update(val)
            else:
                self.__dict__[key] = val


def parse_dict_config(d):
    return ConfigObject(d)


def _print_attributes(obj, indent=0):
    for attr, value in obj.__dict__.items():
        print(" " * indent + f"{attr} = {value}")
        if isinstance(value, object) and hasattr(value, "__dict__"):
            _print_attributes(value, indent + 4)
