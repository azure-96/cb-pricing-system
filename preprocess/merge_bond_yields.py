"""
Merge and preprocess corporate and government bond yields with credit ratings and maturity periods.

This script reads raw yield data and bond metadata from CSV files,
matches company/government bonds with their credit rating and remaining period,
and outputs preprocessed data as pickle files for future use.
"""

import os
import pandas as pd
from config_loader import load_config


def merge_bond_yields() -> pd.DataFrame:
    """
    Load or generate merged bond yield data.

    If preprocessed file exists, it is loaded directly.
    Otherwise, raw CSVs are read and processed into daily bond yield tables.

    Returns:
        pd.DataFrame: Merged corporate bond yield table
        ['TRADE_DT', 'WindID', 'CREDITRATING', 'Period', 'BOND_RATE']
    """
    config = load_config()
    bond_data_path = config['bond_data']
    corp_output_file = os.path.join(bond_data_path, 'bond_rate.pkl')
    gov_output_file = os.path.join(bond_data_path, 'bond_rate_cn.pkl')

    # If preprocessed corporate bond data exists, load and return
    if os.path.exists(corp_output_file):
        bond_rate_daily = pd.read_pickle(corp_output_file)
        print("Loaded existing corporate bond yield data.")
        return bond_rate_daily

    # Otherwise, read raw data and process
    print("Preprocessed file not found. Generating from raw data...")

    bond_yield = pd.read_csv(os.path.join(bond_data_path, 'WIND_ConvertibleBondYield.csv'), encoding='utf-8')
    all_bond_index = pd.read_csv(os.path.join(bond_data_path, 'bonds_index.csv'), encoding='utf-8')

    corp_bonds = all_bond_index[all_bond_index['Type'] == 'Corporate']
    treasury_bonds = all_bond_index[all_bond_index['Type'] == 'Treasury']

    # Merge corporate bond data
    bond_rate_daily = pd.merge(corp_bonds, bond_yield, on='WindID', how='left')
    bond_rate_daily.rename(columns={"TradingDay": "TRADE_DT", "Close": "BOND_RATE"}, inplace=True)
    bond_rate_daily = bond_rate_daily[['TRADE_DT', 'WindID', 'CREDITRATING', 'Period', 'BOND_RATE']]
    bond_rate_daily.to_pickle(corp_output_file)

    # Merge government bond data
    bond_rate_cn = pd.merge(treasury_bonds, bond_yield, on='WindID', how='left')
    bond_rate_cn.rename(columns={"TradingDay": "TRADE_DT", "Close": "BOND_RATE"}, inplace=True)
    bond_rate_cn = bond_rate_cn[['TRADE_DT', 'WindID', 'Period', 'BOND_RATE']]
    bond_rate_cn.to_pickle(gov_output_file)

    print("Bond yield data generated and saved.")
    return bond_rate_daily


if __name__ == '__main__':
    df = merge_bond_yields()
    print("Sample output:")
    print(df.head())
