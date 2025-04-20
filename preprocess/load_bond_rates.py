"""
Merge and preprocess corporate and government bond yields with credit ratings and maturity periods.
"""

import os
import pandas as pd
from config_loader import load_config


def load_bond_rate_table() -> pd.DataFrame:
    """
    Load or generate merged corporate bond yield data with credit ratings and terms.

    Returns:
        pd.DataFrame: Merged corporate bond yield table with columns:
                      ['TRADE_DT', 'WindID', 'CREDITRATING', 'Period', 'BOND_RATE']
    """
    config = load_config()
    bond_data_path = config['bond_data']
    corp_output_file = os.path.join(bond_data_path, 'bond_rate.pkl')
    gov_output_file = os.path.join(bond_data_path, 'bond_rate_cn.pkl')

    if os.path.exists(corp_output_file):
        print("Loaded existing corporate bond yield data.")
        return pd.read_pickle(corp_output_file)

    print("Preprocessed bond yield files not found. Generating from raw data...")

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
    df = load_bond_rate_table()
    print("Sample output:")
    print(df.head())
