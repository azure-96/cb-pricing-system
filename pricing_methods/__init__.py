print("[INFO] pricing_methods module loaded and config initialized.")

from config_loader import load_config

# Initialize and cache the configuration globally
_config = load_config()


# Provide a safe access function to retrieve the config from submodules
def get_config():
    return _config


# Import core pricing functions from submodules
from .bs import bs_cb
from .ccb import ccb_cb
from .mc import mc_cb

# Define public API for the pricing_methods package
__all__ = ["get_config", "bs_cb", "ccb_cb", "mc_cb"]
