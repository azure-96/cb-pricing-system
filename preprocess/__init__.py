print("[INFO] preprocess module loaded.")

from config_loader import load_config

_config = load_config()


# Provide a safe access function to retrieve the config from submodules
def get_config():
    return _config


from preprocess.cb_term_and_rate import get_bond_rate_data, load_cb_terms_dict
from preprocess.compute_cb_implied_vol import compute_cb_implied_vol
from preprocess.compute_stock_volatility import calculate_stock_volatility
from preprocess.generate_cb_terms import load_or_create_cb_terms
from preprocess.load_bond_rates import load_bond_rate_table
from preprocess.merge_bond_yields import merge_bond_yields
from preprocess.merge_cb_stock_data import merge_cb_stock_data

# Define public API
__all__ = [
    "get_bond_rate_data",
    "load_cb_terms_dict",
    "compute_cb_implied_vol",
    "calculate_stock_volatility",
    "load_or_create_cb_terms",
    "load_bond_rate_table",
    "merge_bond_yields",
    "merge_cb_stock_data"
]
