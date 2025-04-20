"""
Main script for preprocessing all convertible bond data before pricing.

This includes: bond rates, CB terms, volatility estimations, and final data merge.
"""
from preprocess.cb_term_and_rate import get_bond_rate_data, load_cb_terms_dict
from preprocess.compute_cb_implied_vol import compute_cb_implied_vol
from preprocess.compute_stock_volatility import calculate_stock_volatility
from preprocess.generate_cb_terms import load_or_create_cb_terms
from preprocess.load_bond_rates import load_bond_rate_table
from preprocess.merge_bond_yields import merge_bond_yields
from preprocess.merge_cb_stock_data import merge_cb_stock_data
from config_loader import load_config


def run_preprocessing():
    config = load_config()
    print("Step 1: Loading bond rate and CB term data.")
    # 1. Bond Rate and CB Term Dictionary
    bond_rate_df = get_bond_rate_data(read_from_file=False)
    cb_term_dict = load_cb_terms_dict(read_from_file=False)
    bond_rate_daily = load_bond_rate_table()

    print("Step 2: Merging bond yield + credit data.")
    bond_rate_merged = merge_bond_yields()

    print("Step 3: Generating CB coupon term dictionary.")
    cb_terms = load_or_create_cb_terms()

    print("Step 4: Computing implied volatility from CB market prices.")
    compute_cb_implied_vol(cb_terms)

    print("Step 5: Computing historical stock volatility.")
    calculate_stock_volatility(config["pricing"]["vol_list"])

    print("Step 6: Merging all data into final CB-stock dataset.")
    merged_df = merge_cb_stock_data(BondRateDaily=bond_rate_merged, CBterms=cb_terms)

    print("All preprocessing completed.")
    print(merged_df.head())


if __name__ == "__main__":
    run_preprocessing()
