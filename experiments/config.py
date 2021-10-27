import ml_collections

def basic_config():
    """Retruns common config for most experiments."""
    config = ml_collections.ConfigDict()
    config.batch_size = 128
    config.lr = 0.03
    config.momentum = 0.9
    config.weight_decay = 0.0004
    return config
    
def get_byol():
    """Returns BYOL configuration."""
    config = basic_config()
    config.name = 'BYOL'
    return config

def get_simsiam():
    """Returns SimSiam configuration."""
    config = basic_config()
    config.name = 'SimSiam'
    return config

def get_direct_pred():
    """Returns DirectPred configuration."""
    config = basic_config()
    config.name = 'DirectPred'
    return config
