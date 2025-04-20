"""
Merge convertible bond data with underlying stock prices and bond rates.

This module prepares a unified DataFrame for each convertible bond,
including its price, terms, interest rates, and stock prices.
"""

import os
import pandas as pd
import dateparser
from config_loader import load_config


def merge_cb_stock_data(BondRateDaily: pd.DataFrame, CBterms: dict) -> pd.DataFrame:
    """
    Merge CB pricing, stock prices, rates, and terms into a single DataFrame.

    Args:
        BondRateDaily (pd.DataFrame): Bond rates from `merge_bond_yields.py`
        CBterms (dict): CB coupon terms from `generate_cb_terms.py`

    Returns:
        pd.DataFrame: Merged panel data, saved as `cb_with_stock_all.pkl`
    """
    config = load_config()
    data_path = config['data_path']
    bond_data_path = config['bond_data']
    stock_data_path = config['stock_data']

    output_file = os.path.join(data_path, 'cb_with_stock_all.pkl')
    if os.path.exists(output_file):
        print("Loaded existing merged CB-stock data.")
        return pd.read_pickle(output_file)

    print("Generating merged CB-stock data from raw files...")

    df_price = pd.read_csv(os.path.join(bond_data_path, 'WIND_ConvertibleBondsDailyQuote.csv'), encoding='utf-8')
    df_clause = pd.read_csv(os.path.join(bond_data_path, 'WIND_ConvertibleBondsAdditionalFields.csv'), encoding='utf-8')
    df_base = pd.read_csv(os.path.join(bond_data_path, 'WIND_ConvertibleBondsBaseInfo.csv'), encoding='utf-8')
    df_ind = pd.read_csv(os.path.join(bond_data_path, 'WIND_ConvertibleBondsIndicator.csv'), encoding='utf-8')
    df_stock = pd.read_csv(os.path.join(stock_data_path, 'UnderlyingStockPrice.csv'), encoding='utf-8')

    df_stock.rename(columns={'S_INFO_WINDCODE': 'UNDERLYINGCODE', 'TRADE_DT': 'TradingDay'}, inplace=True)
    df_stock = df_stock[['TradingDay', 'UNDERLYINGCODE', 'S_DQ_CLOSE']]

    # Get bond maturity (years) from CBterms
    maturity_map = {
        int(code): len(term['Coupon']) for code, term in CBterms.items()
    }
    df_maturity = pd.DataFrame(list(maturity_map.items()), columns=['TRADE_CODE', 'Period'])

    # Merge all bond data
    df_cb = pd.merge(df_price, df_base, on=['TradingDay', 'TRADE_CODE'], how='outer')
    df_cb = pd.merge(df_cb, df_ind, on=['TradingDay', 'TRADE_CODE'], how='left')
    df_cb = pd.merge(df_cb, df_clause, on=['TradingDay', 'TRADE_CODE'], how='left')
    df_cb = pd.merge(df_cb, df_stock, on=['TradingDay', 'UNDERLYINGCODE'], how='left')
    df_cb.rename(columns={'UNDERLYINGCODE': 'S_INFO_WINDCODE', 'TradingDay': 'TRADE_DT'}, inplace=True)

    # Merge bond rate (matched by credit rating & maturity)
    df_cb = pd.merge(df_cb, df_maturity, on='TRADE_CODE', how='left')
    df_rate = df_cb[['TRADE_DT', 'TRADE_CODE', 'CREDITRATING', 'Period']].drop_duplicates()
    df_rate = pd.merge(df_rate, BondRateDaily, on=['TRADE_DT', 'CREDITRATING', 'Period'], how='left')
    df_cb = pd.merge(df_cb, df_rate[['TRADE_DT', 'TRADE_CODE', 'BOND_RATE']], on=['TRADE_DT', 'TRADE_CODE'], how='left')
    df_cb = df_cb[~df_cb['BOND_RATE'].isna()].copy()

    # Convert maturity date string to datetime (expensive operation!)
    df_cb['Maturity'] = df_cb['CLAUSE_CONVERSION_2_SWAPSHAREENDDATE'].apply(dateparser.parse)
    df_cb = df_cb.sort_values(by=['TRADE_CODE', 'TRADE_DT']).reset_index(drop=True)

    df_cb.to_pickle(output_file)
    print("Merged CB-stock data saved.")
    return df_cb


if __name__ == '__main__':
    from merge_bond_yields import merge_bond_yields
    from generate_cb_terms import load_or_create_cb_terms

    bond_rates = merge_bond_yields()
    cb_terms = load_or_create_cb_terms()
    df = merge_cb_stock_data(bond_rates, cb_terms)

    print("Sample output:")
    print(df.head())
