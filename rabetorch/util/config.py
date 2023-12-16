from omegaconf import OmegaConf


def load_config(config_path, override):
    base_cfg = OmegaConf.load(config_path)
    if hasattr(base_cfg, "BASE"):
        for _sub_cfg_path in base_cfg.BASE:
            _sub_cfg = OmegaConf.load("configs/" + _sub_cfg_path)
            base_cfg = OmegaConf.merge(base_cfg, _sub_cfg)

    if hasattr(base_cfg, "PRIMARY_CONFIG"):
        primary_cfg = OmegaConf.load("configs/" + base_cfg.PRIMARY_CONFIG)
        base_cfg = OmegaConf.merge(base_cfg, primary_cfg)

    if override:
        omega_dict = {}
        for key, val in override.items():
            keys = key.split(".")
            _dict = omega_dict
            for k in keys[:-1]:
                if k not in _dict:
                    _dict[k] = {}
                _dict = _dict[k]
            _dict[keys[-1]] = val
        override_cfg = OmegaConf.create(omega_dict)
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    print("Config load has done. Here is confiuration of this run.")
    print(OmegaConf.to_yaml(base_cfg))
    return base_cfg


def smart_type(arg_value):
    """
    Convert the argument to int or float if possible, otherwise return as string.
    """
    try:
        return int(arg_value)
    except ValueError:
        try:
            return float(arg_value)
        except ValueError:
            return arg_value
