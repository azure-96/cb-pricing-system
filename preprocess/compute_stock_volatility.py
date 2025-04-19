# -*- coding: utf-8 -*-
"""Calculate and cache historical volatility for underlying stocks using rolling windows."""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config_loader import load_config


def calculate_stock_volatility(time_windows: list) -> pd.DataFrame:
    """
    Compute or load historical volatility for all underlying stocks over specified rolling windows.

    Args:
        time_windows (list): A list of window sizes (in trading days), e.g., [20, 60, 120, 250]

    Returns:
        pd.DataFrame: Historical volatility dataframe with columns:
                      ['S_INFO_WINDCODE', 'TRADE_DT', 'hist_vol_{window1}', ...]
    """
    config = load_config()
    stock_data_path = config['stock_data']
    output_file = os.path.join(stock_data_path, 'StockHistoricalVolatility.pkl')

    # if os.path.exists(output_file):
    #     print("Loaded existing stock historical volatility data.")
    #     return pd.read_pickle(output_file)

    print("Volatility file not found. Calculating from raw stock price data...")

    df_price = pd.read_csv(os.path.join(stock_data_path, 'UnderlyingStockPrice.csv'), encoding='utf-8')
    result_df = pd.DataFrame()
    grouped = df_price.groupby('S_INFO_WINDCODE')

    for code, group in tqdm(grouped, desc="Computing volatility"):
        group = group.sort_values('TRADE_DT')
        group['ln_return'] = np.log(group['S_DQ_CLOSE'] / group['S_DQ_PRECLOSE'])

        for window in time_windows:
            col_name = f'hist_vol_{window}'
            group[col_name] = group['ln_return'].rolling(window).std() * np.sqrt(252)

        selected_cols = ['S_INFO_WINDCODE', 'TRADE_DT'] + [f'hist_vol_{w}' for w in time_windows]
        result_df = pd.concat([result_df, group[selected_cols]], axis=0)

    result_df.to_pickle(output_file)
    print("Stock volatility calculated and saved.")

    return result_df


if __name__ == '__main__':
    # default_windows = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 250]
    default_windows = [20, 40, 60]
    df = calculate_stock_volatility(default_windows)
    print("Sample output:")
    print(df.head())
