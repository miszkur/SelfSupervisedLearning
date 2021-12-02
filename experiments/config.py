import ml_collections

def basic_config():
    """Retruns common config for most experiments."""
    config = ml_collections.ConfigDict()
    config.num_classes = 10
    config.batch_size = 128
    config.predictor_hidden_size  = 512
    optimizer_params = ml_collections.ConfigDict()
    optimizer_params.lr = 0.03
    optimizer_params.momentum = 0.9
    optimizer_params.weight_decay = 0.0004
    config.optimizer_params = optimizer_params
    # EMA momentum for target network.
    config.tau = 0.996
    config.lambda_ = 0.3 # this is rho in the paper (TODO: change to rho)
    config.symmetry_reg = False
    config.eigenspace_experiment = False 
    config.image_size = (32, 32) # CIFAR10 size.
    config.eps = 0.1
    return config
    
def get_byol():
    """Returns BYOL configuration."""
    config = basic_config()
    config.name = 'BYOL'
    return config

def get_eigenspace_experiment():
    config = get_byol()
    config.eigenspace_experiment = True
    return config

def get_eigenspace_experiment_with_symmetry():
    config = get_byol()
    config.eigenspace_experiment = True
    config.symmetry_reg = True
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
    config.predictor_hidden_size = None
    return config

def get_direct_copy():
    """Returns DirectPred configuration."""
    config = basic_config()
    config.name = 'DirectCopy'
    config.eps = 0.3
    config.lambda_ = 0.5
    config.predictor_hidden_size = None
    # self.gamma = ? # TODO: what is gamma in the paper
    return config

# def get_simclr():
